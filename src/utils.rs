use std::borrow::Cow;

use rustc_ast::ast::{
    self, Attribute, CrateSugar, MetaItem, MetaItemKind, NestedMetaItem, NodeId, Path, Visibility,
    VisibilityKind,
};
use rustc_ast::ptr;
use rustc_ast_pretty::pprust;
use rustc_span::{sym, symbol, BytePos, LocalExpnId, Span, Symbol, SyntaxContext};
use unicode_width::UnicodeWidthStr;

use crate::comment::{filter_normal_code, CharClasses, FullCodeCharKind, LineClasses};
use crate::config::{Config, Version};
use crate::rewrite::RewriteContext;
use crate::shape::{Indent, Shape};

#[inline]
pub(crate) fn depr_skip_annotation() -> Symbol {
    Symbol::intern("rustfmt_skip")
}

#[inline]
pub(crate) fn skip_annotation() -> Symbol {
    Symbol::intern("rustfmt::skip")
}

pub(crate) fn rewrite_ident<'a>(context: &'a RewriteContext<'_>, ident: symbol::Ident) -> &'a str {
    context.snippet(ident.span)
}

// Computes the length of a string's last line, minus offset.
pub(crate) fn extra_offset(text: &str, shape: Shape) -> usize {
    match text.rfind('\n') {
        // 1 for newline character
        Some(idx) => text.len().saturating_sub(idx + 1 + shape.used_width()),
        None => text.len(),
    }
}

pub(crate) fn is_same_visibility(a: &Visibility, b: &Visibility) -> bool {
    match (&a.kind, &b.kind) {
        (
            VisibilityKind::Restricted { path: p, .. },
            VisibilityKind::Restricted { path: q, .. },
        ) => pprust::path_to_string(&p) == pprust::path_to_string(&q),
        (VisibilityKind::Public, VisibilityKind::Public)
        | (VisibilityKind::Inherited, VisibilityKind::Inherited)
        | (
            VisibilityKind::Crate(CrateSugar::PubCrate),
            VisibilityKind::Crate(CrateSugar::PubCrate),
        )
        | (
            VisibilityKind::Crate(CrateSugar::JustCrate),
            VisibilityKind::Crate(CrateSugar::JustCrate),
        ) => true,
        _ => false,
    }
}

// Uses Cow to avoid allocating in the common cases.
pub(crate) fn format_visibility(
    context: &RewriteContext<'_>,
    vis: &Visibility,
) -> Cow<'static, str> {
    match vis.kind {
        VisibilityKind::Public => Cow::from("pub "),
        VisibilityKind::Inherited => Cow::from(""),
        VisibilityKind::Crate(CrateSugar::PubCrate) => Cow::from("pub(crate) "),
        VisibilityKind::Crate(CrateSugar::JustCrate) => Cow::from("crate "),
        VisibilityKind::Restricted { ref path, .. } => {
            let Path { ref segments, .. } = **path;
            let mut segments_iter = segments.iter().map(|seg| rewrite_ident(context, seg.ident));
            if path.is_global() {
                segments_iter
                    .next()
                    .expect("Non-global path in pub(restricted)?");
            }
            let is_keyword = |s: &str| s == "self" || s == "super";
            let path = segments_iter.collect::<Vec<_>>().join("::");
            let in_str = if is_keyword(&path) { "" } else { "in " };

            Cow::from(format!("pub({}{}) ", in_str, path))
        }
    }
}

#[inline]
pub(crate) fn format_async(is_async: &ast::Async) -> &'static str {
    match is_async {
        ast::Async::Yes { .. } => "async ",
        ast::Async::No => "",
    }
}

#[inline]
pub(crate) fn format_constness(constness: ast::Const) -> &'static str {
    match constness {
        ast::Const::Yes(..) => "const ",
        ast::Const::No => "",
    }
}

#[inline]
pub(crate) fn format_constness_right(constness: ast::Const) -> &'static str {
    match constness {
        ast::Const::Yes(..) => " const",
        ast::Const::No => "",
    }
}

#[inline]
pub(crate) fn format_defaultness(defaultness: ast::Defaultness) -> &'static str {
    match defaultness {
        ast::Defaultness::Default(..) => "default ",
        ast::Defaultness::Final => "",
    }
}

#[inline]
pub(crate) fn format_unsafety(unsafety: ast::Unsafe) -> &'static str {
    match unsafety {
        ast::Unsafe::Yes(..) => "unsafe ",
        ast::Unsafe::No => "",
    }
}

#[inline]
pub(crate) fn format_auto(is_auto: ast::IsAuto) -> &'static str {
    match is_auto {
        ast::IsAuto::Yes => "auto ",
        ast::IsAuto::No => "",
    }
}

#[inline]
pub(crate) fn format_mutability(mutability: ast::Mutability) -> &'static str {
    match mutability {
        ast::Mutability::Mut => "mut ",
        ast::Mutability::Not => "",
    }
}

#[inline]
pub(crate) fn format_extern(
    ext: ast::Extern,
    explicit_abi: bool,
    is_mod: bool,
) -> Cow<'static, str> {
    let abi = match ext {
        ast::Extern::None => "Rust".to_owned(),
        ast::Extern::Implicit => "C".to_owned(),
        ast::Extern::Explicit(abi) => abi.symbol_unescaped.to_string(),
    };

    if abi == "Rust" && !is_mod {
        Cow::from("")
    } else if abi == "C" && !explicit_abi {
        Cow::from("extern ")
    } else {
        Cow::from(format!(r#"extern "{}" "#, abi))
    }
}

#[inline]
// Transform `Vec<rustc_ast::ptr::P<T>>` into `Vec<&T>`
pub(crate) fn ptr_vec_to_ref_vec<T>(vec: &[ptr::P<T>]) -> Vec<&T> {
    vec.iter().map(|x| &**x).collect::<Vec<_>>()
}

#[inline]
pub(crate) fn filter_attributes(
    attrs: &[ast::Attribute],
    style: ast::AttrStyle,
) -> Vec<ast::Attribute> {
    attrs
        .iter()
        .filter(|a| a.style == style)
        .cloned()
        .collect::<Vec<_>>()
}

#[inline]
pub(crate) fn inner_attributes(attrs: &[ast::Attribute]) -> Vec<ast::Attribute> {
    filter_attributes(attrs, ast::AttrStyle::Inner)
}

#[inline]
pub(crate) fn outer_attributes(attrs: &[ast::Attribute]) -> Vec<ast::Attribute> {
    filter_attributes(attrs, ast::AttrStyle::Outer)
}

#[inline]
pub(crate) fn is_single_line(s: &str) -> bool {
    !s.chars().any(|c| c == '\n')
}

#[inline]
pub(crate) fn first_line_contains_single_line_comment(s: &str) -> bool {
    s.lines().next().map_or(false, |l| l.contains("//"))
}

#[inline]
pub(crate) fn last_line_contains_single_line_comment(s: &str) -> bool {
    s.lines().last().map_or(false, |l| l.contains("//"))
}

#[inline]
pub(crate) fn is_attributes_extendable(attrs_str: &str) -> bool {
    !attrs_str.contains('\n') && !last_line_contains_single_line_comment(attrs_str)
}

/// The width of the first line in s.
#[inline]
pub(crate) fn first_line_width(s: &str) -> usize {
    unicode_str_width(s.splitn(2, '\n').next().unwrap_or(""))
}

/// The width of the last line in s.
#[inline]
pub(crate) fn last_line_width(s: &str) -> usize {
    unicode_str_width(s.rsplitn(2, '\n').next().unwrap_or(""))
}

/// The total used width of the last line.
#[inline]
pub(crate) fn last_line_used_width(s: &str, offset: usize) -> usize {
    if s.contains('\n') {
        last_line_width(s)
    } else {
        offset + unicode_str_width(s)
    }
}

#[inline]
pub(crate) fn trimmed_last_line_width(s: &str) -> usize {
    unicode_str_width(match s.rfind('\n') {
        Some(n) => s[(n + 1)..].trim(),
        None => s.trim(),
    })
}

#[inline]
pub(crate) fn last_line_extendable(s: &str) -> bool {
    if s.ends_with("\"#") {
        return true;
    }
    for c in s.chars().rev() {
        match c {
            '(' | ')' | ']' | '}' | '?' | '>' => continue,
            '\n' => break,
            _ if c.is_whitespace() => continue,
            _ => return false,
        }
    }
    true
}

#[inline]
fn is_skip(meta_item: &MetaItem) -> bool {
    match meta_item.kind {
        MetaItemKind::Word => {
            let path_str = pprust::path_to_string(&meta_item.path);
            path_str == *skip_annotation().as_str() || path_str == *depr_skip_annotation().as_str()
        }
        MetaItemKind::List(ref l) => {
            meta_item.has_name(sym::cfg_attr) && l.len() == 2 && is_skip_nested(&l[1])
        }
        _ => false,
    }
}

#[inline]
fn is_skip_nested(meta_item: &NestedMetaItem) -> bool {
    match meta_item {
        NestedMetaItem::MetaItem(ref mi) => is_skip(mi),
        NestedMetaItem::Literal(_) => false,
    }
}

#[inline]
pub(crate) fn contains_skip(attrs: &[Attribute]) -> bool {
    attrs
        .iter()
        .any(|a| a.meta().map_or(false, |a| is_skip(&a)))
}

#[inline]
pub(crate) fn semicolon_for_expr(context: &RewriteContext<'_>, expr: &ast::Expr) -> bool {
    // Never try to insert semicolons on expressions when we're inside
    // a macro definition - this can prevent the macro from compiling
    // when used in expression position
    if context.is_macro_def {
        return false;
    }

    match expr.kind {
        ast::ExprKind::Ret(..) | ast::ExprKind::Continue(..) | ast::ExprKind::Break(..) => {
            context.config.trailing_semicolon()
        }
        _ => false,
    }
}

#[inline]
pub(crate) fn semicolon_for_stmt(context: &RewriteContext<'_>, stmt: &ast::Stmt) -> bool {
    match stmt.kind {
        ast::StmtKind::Semi(ref expr) => match expr.kind {
            ast::ExprKind::While(..) | ast::ExprKind::Loop(..) | ast::ExprKind::ForLoop(..) => {
                false
            }
            ast::ExprKind::Break(..) | ast::ExprKind::Continue(..) | ast::ExprKind::Ret(..) => {
                context.config.trailing_semicolon()
            }
            _ => true,
        },
        ast::StmtKind::Expr(..) => false,
        _ => true,
    }
}

#[inline]
pub(crate) fn stmt_expr(stmt: &ast::Stmt) -> Option<&ast::Expr> {
    match stmt.kind {
        ast::StmtKind::Expr(ref expr) => Some(expr),
        _ => None,
    }
}

/// Returns the number of LF and CRLF respectively.
pub(crate) fn count_lf_crlf(input: &str) -> (usize, usize) {
    let mut lf = 0;
    let mut crlf = 0;
    let mut is_crlf = false;
    for c in input.as_bytes() {
        match c {
            b'\r' => is_crlf = true,
            b'\n' if is_crlf => crlf += 1,
            b'\n' => lf += 1,
            _ => is_crlf = false,
        }
    }
    (lf, crlf)
}

pub(crate) fn count_newlines(input: &str) -> usize {
    // Using bytes to omit UTF-8 decoding
    bytecount::count(input.as_bytes(), b'\n')
}

// For format_missing and last_pos, need to use the source callsite (if applicable).
// Required as generated code spans aren't guaranteed to follow on from the last span.
macro_rules! source {
    ($this:ident, $sp:expr) => {
        $sp.source_callsite()
    };
}

pub(crate) fn mk_sp(lo: BytePos, hi: BytePos) -> Span {
    Span::new(lo, hi, SyntaxContext::root(), None)
}

pub(crate) fn mk_sp_lo_plus_one(lo: BytePos) -> Span {
    Span::new(lo, lo + BytePos(1), SyntaxContext::root(), None)
}

// Returns `true` if the given span does not intersect with file lines.
macro_rules! out_of_file_lines_range {
    ($self:ident, $span:expr) => {
        !$self.config.file_lines().is_all()
            && !$self
                .config
                .file_lines()
                .intersects(&$self.parse_sess.lookup_line_range($span))
    };
}

macro_rules! skip_out_of_file_lines_range {
    ($self:ident, $span:expr) => {
        if out_of_file_lines_range!($self, $span) {
            return None;
        }
    };
}

macro_rules! skip_out_of_file_lines_range_visitor {
    ($self:ident, $span:expr) => {
        if out_of_file_lines_range!($self, $span) {
            $self.push_rewrite($span, None);
            return;
        }
    };
}

// Wraps String in an Option. Returns Some when the string adheres to the
// Rewrite constraints defined for the Rewrite trait and None otherwise.
pub(crate) fn wrap_str(s: String, max_width: usize, shape: Shape) -> Option<String> {
    if is_valid_str(&filter_normal_code(&s), max_width, shape) {
        Some(s)
    } else {
        None
    }
}

fn is_valid_str(snippet: &str, max_width: usize, shape: Shape) -> bool {
    if !snippet.is_empty() {
        // First line must fits with `shape.width`.
        if first_line_width(snippet) > shape.width {
            return false;
        }
        // If the snippet does not include newline, we are done.
        if is_single_line(snippet) {
            return true;
        }
        // The other lines must fit within the maximum width.
        if snippet
            .lines()
            .skip(1)
            .any(|line| unicode_str_width(line) > max_width)
        {
            return false;
        }
        // A special check for the last line, since the caller may
        // place trailing characters on this line.
        if last_line_width(snippet) > shape.used_width() + shape.width {
            return false;
        }
    }
    true
}

#[inline]
pub(crate) fn colon_spaces(config: &Config) -> &'static str {
    let before = config.space_before_colon();
    let after = config.space_after_colon();
    match (before, after) {
        (true, true) => " : ",
        (true, false) => " :",
        (false, true) => ": ",
        (false, false) => ":",
    }
}

#[inline]
pub(crate) fn left_most_sub_expr(e: &ast::Expr) -> &ast::Expr {
    match e.kind {
        ast::ExprKind::Call(ref e, _)
        | ast::ExprKind::Binary(_, ref e, _)
        | ast::ExprKind::Cast(ref e, _)
        | ast::ExprKind::Type(ref e, _)
        | ast::ExprKind::Assign(ref e, _, _)
        | ast::ExprKind::AssignOp(_, ref e, _)
        | ast::ExprKind::Field(ref e, _)
        | ast::ExprKind::Index(ref e, _)
        | ast::ExprKind::Range(Some(ref e), _, _)
        | ast::ExprKind::Try(ref e) => left_most_sub_expr(e),
        _ => e,
    }
}

#[inline]
pub(crate) fn starts_with_newline(s: &str) -> bool {
    s.starts_with('\n') || s.starts_with("\r\n")
}

#[inline]
pub(crate) fn first_line_ends_with(s: &str, c: char) -> bool {
    s.lines().next().map_or(false, |l| l.ends_with(c))
}

// States whether an expression's last line exclusively consists of closing
// parens, braces, and brackets in its idiomatic formatting.
pub(crate) fn is_block_expr(context: &RewriteContext<'_>, expr: &ast::Expr, repr: &str) -> bool {
    match expr.kind {
        ast::ExprKind::MacCall(..)
        | ast::ExprKind::Call(..)
        | ast::ExprKind::MethodCall(..)
        | ast::ExprKind::Array(..)
        | ast::ExprKind::Struct(..)
        | ast::ExprKind::While(..)
        | ast::ExprKind::If(..)
        | ast::ExprKind::Block(..)
        | ast::ExprKind::ConstBlock(..)
        | ast::ExprKind::Async(..)
        | ast::ExprKind::Loop(..)
        | ast::ExprKind::ForLoop(..)
        | ast::ExprKind::TryBlock(..)
        | ast::ExprKind::Match(..) => repr.contains('\n'),
        ast::ExprKind::Paren(ref expr)
        | ast::ExprKind::Binary(_, _, ref expr)
        | ast::ExprKind::Index(_, ref expr)
        | ast::ExprKind::Unary(_, ref expr)
        | ast::ExprKind::Closure(_, _, _, _, ref expr, _)
        | ast::ExprKind::Try(ref expr)
        | ast::ExprKind::Yield(Some(ref expr)) => is_block_expr(context, expr, repr),
        // This can only be a string lit
        ast::ExprKind::Lit(_) => {
            repr.contains('\n') && trimmed_last_line_width(repr) <= context.config.tab_spaces()
        }
        ast::ExprKind::AddrOf(..)
        | ast::ExprKind::Assign(..)
        | ast::ExprKind::AssignOp(..)
        | ast::ExprKind::Await(..)
        | ast::ExprKind::Box(..)
        | ast::ExprKind::Break(..)
        | ast::ExprKind::Cast(..)
        | ast::ExprKind::Continue(..)
        | ast::ExprKind::Err
        | ast::ExprKind::Field(..)
        | ast::ExprKind::InlineAsm(..)
        | ast::ExprKind::LlvmInlineAsm(..)
        | ast::ExprKind::Let(..)
        | ast::ExprKind::Path(..)
        | ast::ExprKind::Range(..)
        | ast::ExprKind::Repeat(..)
        | ast::ExprKind::Ret(..)
        | ast::ExprKind::Tup(..)
        | ast::ExprKind::Type(..)
        | ast::ExprKind::Yield(None)
        | ast::ExprKind::Underscore => false,
    }
}

/// Removes trailing spaces from the specified snippet. We do not remove spaces
/// inside strings or comments.
pub(crate) fn remove_trailing_white_spaces(text: &str) -> String {
    let mut buffer = String::with_capacity(text.len());
    let mut space_buffer = String::with_capacity(128);
    for (char_kind, c) in CharClasses::new(text.chars()) {
        match c {
            '\n' => {
                if char_kind == FullCodeCharKind::InString {
                    buffer.push_str(&space_buffer);
                }
                space_buffer.clear();
                buffer.push('\n');
            }
            _ if c.is_whitespace() => {
                space_buffer.push(c);
            }
            _ => {
                if !space_buffer.is_empty() {
                    buffer.push_str(&space_buffer);
                    space_buffer.clear();
                }
                buffer.push(c);
            }
        }
    }
    buffer
}

/// Indent each line according to the specified `indent`.
/// e.g.
///
/// ```rust,compile_fail
/// foo!{
/// x,
/// y,
/// foo(
///     a,
///     b,
///     c,
/// ),
/// }
/// ```
///
/// will become
///
/// ```rust,compile_fail
/// foo!{
///     x,
///     y,
///     foo(
///         a,
///         b,
///         c,
///     ),
/// }
/// ```
pub(crate) fn trim_left_preserve_layout(
    orig: &str,
    indent: Indent,
    config: &Config,
) -> Option<String> {
    let mut lines = LineClasses::new(orig);
    let first_line = lines.next().map(|(_, s)| s.trim_end().to_owned())?;
    let mut trimmed_lines = Vec::with_capacity(16);

    let mut veto_trim = false;
    let min_prefix_space_width = lines
        .filter_map(|(kind, line)| {
            let mut trimmed = true;
            let prefix_space_width = if is_empty_line(&line) {
                None
            } else {
                Some(get_prefix_space_width(config, &line))
            };

            // just InString{Commented} in order to allow the start of a string to be indented
            let new_veto_trim_value = (kind == FullCodeCharKind::InString
                || (config.version() == Version::Two
                    && kind == FullCodeCharKind::InStringCommented))
                && !line.ends_with('\\');
            let line = if veto_trim || new_veto_trim_value {
                veto_trim = new_veto_trim_value;
                trimmed = false;
                line
            } else {
                line.trim().to_owned()
            };
            trimmed_lines.push((trimmed, line, prefix_space_width));

            // Because there is a veto against trimming and indenting lines within a string,
            // such lines should not be taken into account when computing the minimum.
            match kind {
                FullCodeCharKind::InStringCommented | FullCodeCharKind::EndStringCommented
                    if config.version() == Version::Two =>
                {
                    None
                }
                FullCodeCharKind::InString | FullCodeCharKind::EndString => None,
                _ => prefix_space_width,
            }
        })
        .min()?;

    Some(
        first_line
            + "\n"
            + &trimmed_lines
                .iter()
                .map(
                    |&(trimmed, ref line, prefix_space_width)| match prefix_space_width {
                        _ if !trimmed => line.to_owned(),
                        Some(original_indent_width) => {
                            let new_indent_width = indent.width()
                                + original_indent_width.saturating_sub(min_prefix_space_width);
                            let new_indent = Indent::from_width(config, new_indent_width);
                            format!("{}{}", new_indent.to_string(config), line)
                        }
                        None => String::new(),
                    },
                )
                .collect::<Vec<_>>()
                .join("\n"),
    )
}

/// Based on the given line, determine if the next line can be indented or not.
/// This allows to preserve the indentation of multi-line literals.
pub(crate) fn indent_next_line(kind: FullCodeCharKind, _line: &str, config: &Config) -> bool {
    !(kind.is_string() || (config.version() == Version::Two && kind.is_commented_string()))
}

pub(crate) fn is_empty_line(s: &str) -> bool {
    s.is_empty() || s.chars().all(char::is_whitespace)
}

fn get_prefix_space_width(config: &Config, s: &str) -> usize {
    let mut width = 0;
    for c in s.chars() {
        match c {
            ' ' => width += 1,
            '\t' => width += config.tab_spaces(),
            _ => return width,
        }
    }
    width
}

pub(crate) trait NodeIdExt {
    fn root() -> Self;
}

impl NodeIdExt for NodeId {
    fn root() -> NodeId {
        NodeId::placeholder_from_expn_id(LocalExpnId::ROOT)
    }
}

pub(crate) fn unicode_str_width(s: &str) -> usize {
    s.width()
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_remove_trailing_white_spaces() {
        let s = "    r#\"\n        test\n    \"#";
        assert_eq!(remove_trailing_white_spaces(&s), s);
    }

    #[test]
    fn test_trim_left_preserve_layout() {
        let s = "aaa\n\tbbb\n    ccc";
        let config = Config::default();
        let indent = Indent::new(4, 0);
        assert_eq!(
            trim_left_preserve_layout(&s, indent, &config),
            Some("aaa\n    bbb\n    ccc".to_string())
        );
    }
}
