use std::iter;

use rustc_ast::ptr::P;
use rustc_ast::{self as ast, DUMMY_NODE_ID, Expr, ExprKind};
use rustc_ast_pretty::pprust;
use rustc_span::hygiene::{ExpnKind, MacroKind};
use rustc_span::{Span, Symbol, kw, sym};
use smallvec::SmallVec;

use crate::base::{Annotatable, ExtCtxt};
use crate::expand::{AstFragment, AstFragmentKind};

#[derive(Default)]
pub struct MacroStat {
    /// Number of uses of the macro.
    pub uses: usize,

    /// Number of lines of code (when pretty-printed).
    pub lines: usize,

    /// Number of bytes of code (when pretty-printed).
    pub bytes: usize,
}

pub(crate) fn elems_to_string<T>(elems: &SmallVec<[T; 1]>, f: impl Fn(&T) -> String) -> String {
    let mut s = String::new();
    for (i, elem) in elems.iter().enumerate() {
        if i > 0 {
            s.push('\n');
        }
        s.push_str(&f(elem));
    }
    s
}

pub(crate) fn unreachable_to_string<T>(_: &T) -> String {
    unreachable!()
}

pub(crate) fn update_bang_macro_stats(
    ecx: &mut ExtCtxt<'_>,
    fragment_kind: AstFragmentKind,
    span: Span,
    mac: P<ast::MacCall>,
    fragment: &AstFragment,
) {
    // Does this path match any of the include macros, e.g. `include!`?
    // Ignore them. They would have large numbers but are entirely
    // unsurprising and uninteresting.
    let is_include_path = mac.path == sym::include
        || mac.path == sym::include_bytes
        || mac.path == sym::include_str
        || mac.path == [sym::std, sym::include].as_slice() // std::include
        || mac.path == [sym::std, sym::include_bytes].as_slice() // std::include_bytes
        || mac.path == [sym::std, sym::include_str].as_slice(); // std::include_str
    if is_include_path {
        return;
    }

    // The call itself (e.g. `println!("hi")`) is the input. Need to wrap
    // `mac` in something printable; `ast::Expr` is as good as anything
    // else.
    let expr = Expr {
        id: DUMMY_NODE_ID,
        kind: ExprKind::MacCall(mac),
        span: Default::default(),
        attrs: Default::default(),
        tokens: None,
    };
    let input = pprust::expr_to_string(&expr);

    // Get `mac` back out of `expr`.
    let ast::Expr { kind: ExprKind::MacCall(mac), .. } = expr else { unreachable!() };

    update_macro_stats(ecx, MacroKind::Bang, fragment_kind, span, &mac.path, &input, fragment);
}

pub(crate) fn update_attr_macro_stats(
    ecx: &mut ExtCtxt<'_>,
    fragment_kind: AstFragmentKind,
    span: Span,
    path: &ast::Path,
    attr: &ast::Attribute,
    item: Annotatable,
    fragment: &AstFragment,
) {
    // Does this path match `#[derive(...)]` in any of its forms? If so,
    // ignore it because the individual derives will go through the
    // `Invocation::Derive` handling separately.
    let is_derive_path = *path == sym::derive
        // ::core::prelude::v1::derive
        || *path == [kw::PathRoot, sym::core, sym::prelude, sym::v1, sym::derive].as_slice();
    if is_derive_path {
        return;
    }

    // The attribute plus the item itself constitute the input, which we
    // measure.
    let input = format!(
        "{}\n{}",
        pprust::attribute_to_string(attr),
        fragment_kind.expect_from_annotatables(iter::once(item)).to_string(),
    );
    update_macro_stats(ecx, MacroKind::Attr, fragment_kind, span, path, &input, fragment);
}

pub(crate) fn update_derive_macro_stats(
    ecx: &mut ExtCtxt<'_>,
    fragment_kind: AstFragmentKind,
    span: Span,
    path: &ast::Path,
    fragment: &AstFragment,
) {
    // Use something like `#[derive(Clone)]` for the measured input, even
    // though it may have actually appeared in a multi-derive attribute
    // like `#[derive(Clone, Copy, Debug)]`.
    let input = format!("#[derive({})]", pprust::path_to_string(path));
    update_macro_stats(ecx, MacroKind::Derive, fragment_kind, span, path, &input, fragment);
}

pub(crate) fn update_macro_stats(
    ecx: &mut ExtCtxt<'_>,
    macro_kind: MacroKind,
    fragment_kind: AstFragmentKind,
    span: Span,
    path: &ast::Path,
    input: &str,
    fragment: &AstFragment,
) {
    // Measure the size of the output by pretty-printing it and counting
    // the lines and bytes.
    let name = Symbol::intern(&pprust::path_to_string(path));
    let output = fragment.to_string();
    let num_lines = output.trim_end().split('\n').count();
    let num_bytes = output.len();

    // This code is useful for debugging `-Zmacro-stats`. For every
    // invocation it prints the full input and output.
    if false {
        let name = ExpnKind::Macro(macro_kind, name).descr();
        let crate_name = &ecx.ecfg.crate_name;
        let span = ecx
            .sess
            .source_map()
            .span_to_string(span, rustc_span::FileNameDisplayPreference::Local);
        eprint!(
            "\
            -------------------------------\n\
            {name}: [{crate_name}] ({fragment_kind:?}) {span}\n\
            -------------------------------\n\
            {input}\n\
            -- {num_lines} lines, {num_bytes} bytes --\n\
            {output}\n\
        "
        );
    }

    // The recorded size is the difference between the input and the output.
    let entry = ecx.macro_stats.entry((name, macro_kind)).or_insert(MacroStat::default());
    entry.uses += 1;
    entry.lines += num_lines;
    entry.bytes += num_bytes;
}
