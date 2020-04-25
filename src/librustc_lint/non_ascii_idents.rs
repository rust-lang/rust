use crate::{EarlyContext, EarlyLintPass, LintContext};
use rustc_ast::ast;
use rustc_data_structures::fx::FxHashMap;
use rustc_span::symbol::SymbolStr;

declare_lint! {
    pub NON_ASCII_IDENTS,
    Allow,
    "detects non-ASCII identifiers"
}

declare_lint! {
    pub UNCOMMON_CODEPOINTS,
    Warn,
    "detects uncommon Unicode codepoints in identifiers"
}

// FIXME: Change this to warn.
declare_lint! {
    pub CONFUSABLE_IDENTS,
    Allow,
    "detects visually confusable pairs between identifiers"
}

declare_lint_pass!(NonAsciiIdents => [NON_ASCII_IDENTS, UNCOMMON_CODEPOINTS, CONFUSABLE_IDENTS]);

use std::hash::{Hash, Hasher};
use std::ops::Deref;

enum CowBoxSymStr {
    Interned(SymbolStr),
    Owned(Box<str>),
}

impl Deref for CowBoxSymStr {
    type Target = str;

    fn deref(&self) -> &str {
        match self {
            CowBoxSymStr::Interned(interned) => interned,
            CowBoxSymStr::Owned(ref owned) => owned,
        }
    }
}

impl Hash for CowBoxSymStr {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        Hash::hash(&**self, state)
    }
}

impl PartialEq<CowBoxSymStr> for CowBoxSymStr {
    #[inline]
    fn eq(&self, other: &CowBoxSymStr) -> bool {
        PartialEq::eq(&**self, &**other)
    }
}

impl Eq for CowBoxSymStr {}

fn calc_skeleton(symbol_str: SymbolStr, buffer: &'_ mut String) -> CowBoxSymStr {
    use std::mem::swap;
    use unicode_security::confusable_detection::skeleton;
    buffer.clear();
    buffer.extend(skeleton(&symbol_str));
    if symbol_str == *buffer {
        CowBoxSymStr::Interned(symbol_str)
    } else {
        let mut owned = String::new();
        swap(buffer, &mut owned);
        CowBoxSymStr::Owned(owned.into_boxed_str())
    }
}

impl EarlyLintPass for NonAsciiIdents {
    fn check_crate(&mut self, cx: &EarlyContext<'_>, _: &ast::Crate) {
        use rustc_session::lint::Level;
        if cx.builder.lint_level(CONFUSABLE_IDENTS).0 == Level::Allow {
            return;
        }
        let symbols = cx.sess.parse_sess.symbol_gallery.symbols.lock();
        let mut symbol_strs_and_spans = Vec::with_capacity(symbols.len());
        for (symbol, sp) in symbols.iter() {
            let symbol_str = symbol.as_str();
            symbol_strs_and_spans.push((symbol_str, *sp));
        }
        drop(symbols);
        symbol_strs_and_spans.sort_by_key(|x| x.0.clone());
        let mut skeleton_map =
            FxHashMap::with_capacity_and_hasher(symbol_strs_and_spans.len(), Default::default());
        let mut str_buf = String::new();
        for (symbol_str, sp) in symbol_strs_and_spans {
            let skeleton = calc_skeleton(symbol_str.clone(), &mut str_buf);
            skeleton_map
                .entry(skeleton)
                .and_modify(|(existing_symbolstr, existing_span)| {
                    cx.struct_span_lint(CONFUSABLE_IDENTS, sp, |lint| {
                        lint.build(&format!(
                            "identifier pair considered confusable between `{}` and `{}`",
                            existing_symbolstr, symbol_str
                        ))
                        .span_label(
                            *existing_span,
                            "this is where the previous identifier occurred",
                        )
                        .emit();
                    });
                })
                .or_insert((symbol_str, sp));
        }
    }
    fn check_ident(&mut self, cx: &EarlyContext<'_>, ident: ast::Ident) {
        use unicode_security::GeneralSecurityProfile;
        let name_str = ident.name.as_str();
        if name_str.is_ascii() {
            return;
        }
        cx.struct_span_lint(NON_ASCII_IDENTS, ident.span, |lint| {
            lint.build("identifier contains non-ASCII characters").emit()
        });
        if !name_str.chars().all(GeneralSecurityProfile::identifier_allowed) {
            cx.struct_span_lint(UNCOMMON_CODEPOINTS, ident.span, |lint| {
                lint.build("identifier contains uncommon Unicode codepoints").emit()
            })
        }
    }
}
