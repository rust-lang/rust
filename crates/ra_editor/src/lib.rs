mod code_actions;
mod extend_selection;
mod folding_ranges;
mod line_index;
mod line_index_utils;
mod symbols;
#[cfg(test)]
mod test_utils;
mod typing;

pub use self::{
    code_actions::{add_derive, add_impl, flip_comma, introduce_variable, make_pub_crate, LocalEdit},
    extend_selection::extend_selection,
    folding_ranges::{folding_ranges, Fold, FoldKind},
    line_index::{LineCol, LineIndex},
    line_index_utils::translate_offset_with_edit,
    symbols::{file_structure, file_symbols, FileSymbol, StructureNode},
    typing::{join_lines, on_enter, on_eq_typed},
};
use ra_text_edit::{TextEdit, TextEditBuilder};
use ra_syntax::{
    algo::find_leaf_at_offset,
    ast::{self, AstNode, NameOwner},
    SourceFileNode,
    Location,
    SyntaxKind::{self, *},
    SyntaxNodeRef, TextRange, TextUnit, Direction,
};
use itertools::Itertools;
use rustc_hash::FxHashSet;

#[derive(Debug)]
pub struct HighlightedRange {
    pub range: TextRange,
    pub tag: &'static str,
}

#[derive(Debug, Copy, Clone)]
pub enum Severity {
    Error,
    WeakWarning,
}

#[derive(Debug)]
pub struct Diagnostic {
    pub range: TextRange,
    pub msg: String,
    pub severity: Severity,
    pub fix: Option<LocalEdit>,
}

#[derive(Debug)]
pub struct Runnable {
    pub range: TextRange,
    pub kind: RunnableKind,
}

#[derive(Debug)]
pub enum RunnableKind {
    Test { name: String },
    Bin,
}

pub fn matching_brace(file: &SourceFileNode, offset: TextUnit) -> Option<TextUnit> {
    const BRACES: &[SyntaxKind] = &[
        L_CURLY, R_CURLY, L_BRACK, R_BRACK, L_PAREN, R_PAREN, L_ANGLE, R_ANGLE,
    ];
    let (brace_node, brace_idx) = find_leaf_at_offset(file.syntax(), offset)
        .filter_map(|node| {
            let idx = BRACES.iter().position(|&brace| brace == node.kind())?;
            Some((node, idx))
        })
        .next()?;
    let parent = brace_node.parent()?;
    let matching_kind = BRACES[brace_idx ^ 1];
    let matching_node = parent
        .children()
        .find(|node| node.kind() == matching_kind)?;
    Some(matching_node.range().start())
}

pub fn highlight(file: &SourceFileNode) -> Vec<HighlightedRange> {
    // Visited nodes to handle highlighting priorities
    let mut highlighted = FxHashSet::default();
    let mut res = Vec::new();
    for node in file.syntax().descendants() {
        if highlighted.contains(&node) {
            continue;
        }
        let tag = match node.kind() {
            COMMENT => "comment",
            STRING | RAW_STRING | RAW_BYTE_STRING | BYTE_STRING => "string",
            ATTR => "attribute",
            NAME_REF => "text",
            NAME => "function",
            INT_NUMBER | FLOAT_NUMBER | CHAR | BYTE => "literal",
            LIFETIME => "parameter",
            k if k.is_keyword() => "keyword",
            _ => {
                if let Some(macro_call) = ast::MacroCall::cast(node) {
                    if let Some(path) = macro_call.path() {
                        if let Some(segment) = path.segment() {
                            if let Some(name_ref) = segment.name_ref() {
                                highlighted.insert(name_ref.syntax());
                                let range_start = name_ref.syntax().range().start();
                                let mut range_end = name_ref.syntax().range().end();
                                for sibling in path.syntax().siblings(Direction::Next) {
                                    match sibling.kind() {
                                        EXCL | IDENT => range_end = sibling.range().end(),
                                        _ => (),
                                    }
                                }
                                res.push(HighlightedRange {
                                    range: TextRange::from_to(range_start, range_end),
                                    tag: "macro",
                                })
                            }
                        }
                    }
                }
                continue;
            }
        };
        res.push(HighlightedRange {
            range: node.range(),
            tag,
        })
    }
    res
}

pub fn diagnostics(file: &SourceFileNode) -> Vec<Diagnostic> {
    fn location_to_range(location: Location) -> TextRange {
        match location {
            Location::Offset(offset) => TextRange::offset_len(offset, 1.into()),
            Location::Range(range) => range,
        }
    }

    let mut errors: Vec<Diagnostic> = file
        .errors()
        .into_iter()
        .map(|err| Diagnostic {
            range: location_to_range(err.location()),
            msg: format!("Syntax Error: {}", err),
            severity: Severity::Error,
            fix: None,
        })
        .collect();

    let warnings = check_unnecessary_braces_in_use_statement(file);

    errors.extend(warnings);
    errors
}

fn check_unnecessary_braces_in_use_statement(file: &SourceFileNode) -> Vec<Diagnostic> {
    let mut diagnostics = Vec::new();
    for node in file.syntax().descendants() {
        if let Some(use_tree_list) = ast::UseTreeList::cast(node) {
            if let Some((single_use_tree,)) = use_tree_list.use_trees().collect_tuple() {
                let range = use_tree_list.syntax().range();
                let edit = text_edit_for_remove_unnecessary_braces_with_self_in_use_statement(
                    single_use_tree,
                )
                .unwrap_or_else(|| {
                    let to_replace = single_use_tree.syntax().text().to_string();
                    let mut edit_builder = TextEditBuilder::new();
                    edit_builder.delete(range);
                    edit_builder.insert(range.start(), to_replace);
                    edit_builder.finish()
                });

                diagnostics.push(Diagnostic {
                    range: range,
                    msg: format!("Unnecessary braces in use statement"),
                    severity: Severity::WeakWarning,
                    fix: Some(LocalEdit {
                        label: "Remove unnecessary braces".to_string(),
                        edit: edit,
                        cursor_position: None,
                    }),
                })
            }
        }
    }

    diagnostics
}

fn text_edit_for_remove_unnecessary_braces_with_self_in_use_statement(
    single_use_tree: ast::UseTree,
) -> Option<TextEdit> {
    let use_tree_list_node = single_use_tree.syntax().parent()?;
    if single_use_tree
        .path()?
        .segment()?
        .syntax()
        .first_child()?
        .kind()
        == SyntaxKind::SELF_KW
    {
        let start = use_tree_list_node.prev_sibling()?.range().start();
        let end = use_tree_list_node.range().end();
        let range = TextRange::from_to(start, end);
        let mut edit_builder = TextEditBuilder::new();
        edit_builder.delete(range);
        return Some(edit_builder.finish());
    }
    None
}

pub fn syntax_tree(file: &SourceFileNode) -> String {
    ::ra_syntax::utils::dump_tree(file.syntax())
}

pub fn runnables(file: &SourceFileNode) -> Vec<Runnable> {
    file.syntax()
        .descendants()
        .filter_map(ast::FnDef::cast)
        .filter_map(|f| {
            let name = f.name()?.text();
            let kind = if name == "main" {
                RunnableKind::Bin
            } else if f.has_atom_attr("test") {
                RunnableKind::Test {
                    name: name.to_string(),
                }
            } else {
                return None;
            };
            Some(Runnable {
                range: f.syntax().range(),
                kind,
            })
        })
        .collect()
}

pub fn find_node_at_offset<'a, N: AstNode<'a>>(
    syntax: SyntaxNodeRef<'a>,
    offset: TextUnit,
) -> Option<N> {
    find_leaf_at_offset(syntax, offset).find_map(|leaf| leaf.ancestors().find_map(N::cast))
}

#[cfg(test)]
mod tests {
    use crate::test_utils::{add_cursor, assert_eq_dbg, assert_eq_text, extract_offset};

    use super::*;

    #[test]
    fn test_highlighting() {
        let file = SourceFileNode::parse(
            r#"
// comment
fn main() {}
    println!("Hello, {}!", 92);
"#,
        );
        let hls = highlight(&file);
        assert_eq_dbg(
            r#"[HighlightedRange { range: [1; 11), tag: "comment" },
                HighlightedRange { range: [12; 14), tag: "keyword" },
                HighlightedRange { range: [15; 19), tag: "function" },
                HighlightedRange { range: [29; 37), tag: "macro" },
                HighlightedRange { range: [38; 50), tag: "string" },
                HighlightedRange { range: [52; 54), tag: "literal" }]"#,
            &hls,
        );
    }

    #[test]
    fn test_runnables() {
        let file = SourceFileNode::parse(
            r#"
fn main() {}

#[test]
fn test_foo() {}

#[test]
#[ignore]
fn test_foo() {}
"#,
        );
        let runnables = runnables(&file);
        assert_eq_dbg(
            r#"[Runnable { range: [1; 13), kind: Bin },
                Runnable { range: [15; 39), kind: Test { name: "test_foo" } },
                Runnable { range: [41; 75), kind: Test { name: "test_foo" } }]"#,
            &runnables,
        )
    }

    #[test]
    fn test_matching_brace() {
        fn do_check(before: &str, after: &str) {
            let (pos, before) = extract_offset(before);
            let file = SourceFileNode::parse(&before);
            let new_pos = match matching_brace(&file, pos) {
                None => pos,
                Some(pos) => pos,
            };
            let actual = add_cursor(&before, new_pos);
            assert_eq_text!(after, &actual);
        }

        do_check("struct Foo { a: i32, }<|>", "struct Foo <|>{ a: i32, }");
    }

    #[test]
    fn test_check_unnecessary_braces_in_use_statement() {
        fn check_not_applicable(code: &str) {
            let file = SourceFileNode::parse(code);
            let diagnostics = check_unnecessary_braces_in_use_statement(&file);
            assert!(diagnostics.is_empty());
        }

        fn check_apply(before: &str, after: &str) {
            let file = SourceFileNode::parse(before);
            let diagnostic = check_unnecessary_braces_in_use_statement(&file)
                .pop()
                .unwrap_or_else(|| panic!("no diagnostics for:\n{}\n", before));
            let fix = diagnostic.fix.unwrap();
            let actual = fix.edit.apply(&before);
            assert_eq_text!(after, &actual);
        }

        check_not_applicable(
            "
            use a;
            use a::{c, d::e};
        ",
        );
        check_apply("use {b};", "use b;");
        check_apply("use a::{c};", "use a::c;");
        check_apply("use a::{self};", "use a;");
        check_apply("use a::{c, d::{e}};", "use a::{c, d::e};");
    }
}
