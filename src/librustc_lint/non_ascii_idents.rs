use crate::{EarlyContext, EarlyLintPass, LintContext};
use rustc_ast::ast;
use rustc_data_structures::fx::FxHashMap;
use rustc_span::symbol::SymbolStr;

declare_lint! {
    pub NON_ASCII_IDENTS,
    Allow,
    "detects non-ASCII identifiers",
    crate_level_only
}

declare_lint! {
    pub UNCOMMON_CODEPOINTS,
    Warn,
    "detects uncommon Unicode codepoints in identifiers",
    crate_level_only
}

declare_lint! {
    pub CONFUSABLE_IDENTS,
    Warn,
    "detects visually confusable pairs between identifiers",
    crate_level_only
}

declare_lint_pass!(NonAsciiIdents => [NON_ASCII_IDENTS, UNCOMMON_CODEPOINTS, CONFUSABLE_IDENTS]);

impl EarlyLintPass for NonAsciiIdents {
    fn check_crate(&mut self, cx: &EarlyContext<'_>, _: &ast::Crate) {
        use rustc_session::lint::Level;
        use rustc_span::Span;
        use unicode_security::GeneralSecurityProfile;
        use utils::CowBoxSymStr;

        let check_non_ascii_idents = cx.builder.lint_level(NON_ASCII_IDENTS).0 != Level::Allow;
        let check_uncommon_codepoints =
            cx.builder.lint_level(UNCOMMON_CODEPOINTS).0 != Level::Allow;
        let check_confusable_idents = cx.builder.lint_level(CONFUSABLE_IDENTS).0 != Level::Allow;

        if !check_non_ascii_idents && !check_uncommon_codepoints && !check_confusable_idents {
            return;
        }

        let mut has_non_ascii_idents = false;
        let symbols = cx.sess.parse_sess.symbol_gallery.symbols.lock();
        for (symbol, &sp) in symbols.iter() {
            let symbol_str = symbol.as_str();
            if symbol_str.is_ascii() {
                continue;
            }
            has_non_ascii_idents = true;
            cx.struct_span_lint(NON_ASCII_IDENTS, sp, |lint| {
                lint.build("identifier contains non-ASCII characters").emit()
            });
            if check_uncommon_codepoints
                && !symbol_str.chars().all(GeneralSecurityProfile::identifier_allowed)
            {
                cx.struct_span_lint(UNCOMMON_CODEPOINTS, sp, |lint| {
                    lint.build("identifier contains uncommon Unicode codepoints").emit()
                })
            }
        }

        if has_non_ascii_idents && check_confusable_idents {
            let mut skeleton_map: FxHashMap<CowBoxSymStr, (SymbolStr, Span, bool)> =
                FxHashMap::with_capacity_and_hasher(symbols.len(), Default::default());
            let mut str_buf = String::new();
            for (symbol, &sp) in symbols.iter() {
                fn calc_skeleton(symbol_str: &SymbolStr, buffer: &mut String) -> CowBoxSymStr {
                    use std::mem::replace;
                    use unicode_security::confusable_detection::skeleton;
                    buffer.clear();
                    buffer.extend(skeleton(symbol_str));
                    if *symbol_str == *buffer {
                        CowBoxSymStr::Interned(symbol_str.clone())
                    } else {
                        let owned = replace(buffer, String::new());
                        CowBoxSymStr::Owned(owned.into_boxed_str())
                    }
                }
                let symbol_str = symbol.as_str();
                let is_ascii = symbol_str.is_ascii();
                let skeleton = calc_skeleton(&symbol_str, &mut str_buf);
                skeleton_map
                    .entry(skeleton)
                    .and_modify(|(existing_symbolstr, existing_span, existing_is_ascii)| {
                        if !*existing_is_ascii || !is_ascii {
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
                        }
                        if *existing_is_ascii && !is_ascii {
                            *existing_symbolstr = symbol_str.clone();
                            *existing_span = sp;
                            *existing_is_ascii = is_ascii;
                        }
                    })
                    .or_insert((symbol_str, sp, is_ascii));
            }
        }
    }
}

mod utils {
    use rustc_span::symbol::SymbolStr;
    use std::hash::{Hash, Hasher};
    use std::ops::Deref;

    pub(super) enum CowBoxSymStr {
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
}
