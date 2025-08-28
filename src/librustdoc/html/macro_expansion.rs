use rustc_ast::visit::{Visitor, walk_crate, walk_expr, walk_item, walk_pat, walk_stmt};
use rustc_ast::{Crate, Expr, Item, Pat, Stmt};
use rustc_data_structures::fx::FxHashMap;
use rustc_span::source_map::SourceMap;
use rustc_span::{BytePos, Span};

use crate::config::{OutputFormat, RenderOptions};

/// It returns the expanded macros correspondence map.
pub(crate) fn source_macro_expansion(
    krate: &Crate,
    render_options: &RenderOptions,
    output_format: OutputFormat,
    source_map: &SourceMap,
) -> FxHashMap<BytePos, Vec<ExpandedCode>> {
    if output_format == OutputFormat::Html
        && !render_options.html_no_source
        && render_options.generate_macro_expansion
    {
        let mut expanded_visitor = ExpandedCodeVisitor { expanded_codes: Vec::new(), source_map };
        walk_crate(&mut expanded_visitor, krate);
        expanded_visitor.compute_expanded()
    } else {
        Default::default()
    }
}

/// Contains information about macro expansion in the source code pages.
#[derive(Debug)]
pub(crate) struct ExpandedCode {
    /// The line where the macro expansion starts.
    pub(crate) start_line: u32,
    /// The line where the macro expansion ends.
    pub(crate) end_line: u32,
    /// The source code of the expanded macro.
    pub(crate) code: String,
    /// The span of macro callsite.
    pub(crate) span: Span,
}

/// Contains temporary information of macro expanded code.
///
/// As we go through the HIR visitor, if any span overlaps with another, they will
/// both be merged.
struct ExpandedCodeInfo {
    /// Callsite of the macro.
    span: Span,
    /// Expanded macro source code (HTML escaped).
    code: String,
    /// Span of macro-generated code.
    expanded_span: Span,
}

/// HIR visitor which retrieves expanded macro.
///
/// Once done, the `expanded_codes` will be transformed into a vec of [`ExpandedCode`]
/// which contains the information needed when running the source code highlighter.
pub(crate) struct ExpandedCodeVisitor<'ast> {
    expanded_codes: Vec<ExpandedCodeInfo>,
    source_map: &'ast SourceMap,
}

impl<'ast> ExpandedCodeVisitor<'ast> {
    fn handle_new_span<F: Fn() -> String>(&mut self, new_span: Span, f: F) {
        if new_span.is_dummy() || !new_span.from_expansion() {
            return;
        }
        let callsite_span = new_span.source_callsite();
        if let Some(index) =
            self.expanded_codes.iter().position(|info| info.span.overlaps(callsite_span))
        {
            let info = &mut self.expanded_codes[index];
            if new_span.contains(info.expanded_span) {
                // New macro expansion recursively contains the old one, so replace it.
                info.span = callsite_span;
                info.expanded_span = new_span;
                info.code = f();
            } else {
                // We push the new item after the existing one.
                let expanded_code = &mut self.expanded_codes[index];
                expanded_code.code.push('\n');
                expanded_code.code.push_str(&f());
                let lo = BytePos(expanded_code.expanded_span.lo().0.min(new_span.lo().0));
                let hi = BytePos(expanded_code.expanded_span.hi().0.min(new_span.hi().0));
                expanded_code.expanded_span = expanded_code.expanded_span.with_lo(lo).with_hi(hi);
            }
        } else {
            // We add a new item.
            self.expanded_codes.push(ExpandedCodeInfo {
                span: callsite_span,
                code: f(),
                expanded_span: new_span,
            });
        }
    }

    fn compute_expanded(mut self) -> FxHashMap<BytePos, Vec<ExpandedCode>> {
        self.expanded_codes.sort_unstable_by(|item1, item2| item1.span.cmp(&item2.span));
        let mut expanded: FxHashMap<BytePos, Vec<ExpandedCode>> = FxHashMap::default();
        for ExpandedCodeInfo { span, code, .. } in self.expanded_codes {
            if let Ok(lines) = self.source_map.span_to_lines(span)
                && !lines.lines.is_empty()
            {
                let mut out = String::new();
                super::highlight::write_code(&mut out, &code, None, None, None);
                let first = lines.lines.first().unwrap();
                let end = lines.lines.last().unwrap();
                expanded.entry(lines.file.start_pos).or_default().push(ExpandedCode {
                    start_line: first.line_index as u32 + 1,
                    end_line: end.line_index as u32 + 1,
                    code: out,
                    span,
                });
            }
        }
        expanded
    }
}

// We need to use the AST pretty printing because:
//
// 1. HIR pretty printing doesn't display accurately the code (like `impl Trait`).
// 2. `SourceMap::snippet_opt` might fail if the source is not available.
impl<'ast> Visitor<'ast> for ExpandedCodeVisitor<'ast> {
    fn visit_expr(&mut self, expr: &'ast Expr) {
        if expr.span.from_expansion() {
            self.handle_new_span(expr.span, || rustc_ast_pretty::pprust::expr_to_string(expr));
        } else {
            walk_expr(self, expr);
        }
    }

    fn visit_item(&mut self, item: &'ast Item) {
        if item.span.from_expansion() {
            self.handle_new_span(item.span, || rustc_ast_pretty::pprust::item_to_string(item));
        } else {
            walk_item(self, item);
        }
    }

    fn visit_stmt(&mut self, stmt: &'ast Stmt) {
        if stmt.span.from_expansion() {
            self.handle_new_span(stmt.span, || rustc_ast_pretty::pprust::stmt_to_string(stmt));
        } else {
            walk_stmt(self, stmt);
        }
    }

    fn visit_pat(&mut self, pat: &'ast Pat) {
        if pat.span.from_expansion() {
            self.handle_new_span(pat.span, || rustc_ast_pretty::pprust::pat_to_string(pat));
        } else {
            walk_pat(self, pat);
        }
    }
}
