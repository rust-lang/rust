use crate::{EarlyContext, EarlyLintPass, LintContext};
use rustc_ast as ast;
use rustc_data_structures::fx::FxHashMap;
use rustc_span::symbol::Symbol;

declare_lint! {
    /// The `non_ascii_idents` lint detects non-ASCII identifiers.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// # #![allow(unused)]
    /// #![deny(non_ascii_idents)]
    /// fn main() {
    ///     let föö = 1;
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// This lint allows projects that wish to retain the limit of only using
    /// ASCII characters to switch this lint to "forbid" (for example to ease
    /// collaboration or for security reasons).
    /// See [RFC 2457] for more details.
    ///
    /// [RFC 2457]: https://github.com/rust-lang/rfcs/blob/master/text/2457-non-ascii-idents.md
    pub NON_ASCII_IDENTS,
    Allow,
    "detects non-ASCII identifiers",
    crate_level_only
}

declare_lint! {
    /// The `uncommon_codepoints` lint detects uncommon Unicode codepoints in
    /// identifiers.
    ///
    /// ### Example
    ///
    /// ```rust
    /// # #![allow(unused)]
    /// const µ: f64 = 0.000001;
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// This lint warns about using characters which are not commonly used, and may
    /// cause visual confusion.
    ///
    /// This lint is triggered by identifiers that contain a codepoint that is
    /// not part of the set of "Allowed" codepoints as described by [Unicode®
    /// Technical Standard #39 Unicode Security Mechanisms Section 3.1 General
    /// Security Profile for Identifiers][TR39Allowed].
    ///
    /// Note that the set of uncommon codepoints may change over time. Beware
    /// that if you "forbid" this lint that existing code may fail in the
    /// future.
    ///
    /// [TR39Allowed]: https://www.unicode.org/reports/tr39/#General_Security_Profile
    pub UNCOMMON_CODEPOINTS,
    Warn,
    "detects uncommon Unicode codepoints in identifiers",
    crate_level_only
}

declare_lint! {
    /// The `confusable_idents` lint detects visually confusable pairs between
    /// identifiers.
    ///
    /// ### Example
    ///
    /// ```rust
    /// // Latin Capital Letter E With Caron
    /// pub const Ě: i32 = 1;
    /// // Latin Capital Letter E With Breve
    /// pub const Ĕ: i32 = 2;
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// This lint warns when different identifiers may appear visually similar,
    /// which can cause confusion.
    ///
    /// The confusable detection algorithm is based on [Unicode® Technical
    /// Standard #39 Unicode Security Mechanisms Section 4 Confusable
    /// Detection][TR39Confusable]. For every distinct identifier X execute
    /// the function `skeleton(X)`. If there exist two distinct identifiers X
    /// and Y in the same crate where `skeleton(X) = skeleton(Y)` report it.
    /// The compiler uses the same mechanism to check if an identifier is too
    /// similar to a keyword.
    ///
    /// Note that the set of confusable characters may change over time.
    /// Beware that if you "forbid" this lint that existing code may fail in
    /// the future.
    ///
    /// [TR39Confusable]: https://www.unicode.org/reports/tr39/#Confusable_Detection
    pub CONFUSABLE_IDENTS,
    Warn,
    "detects visually confusable pairs between identifiers",
    crate_level_only
}

declare_lint! {
    /// The `mixed_script_confusables` lint detects visually confusable
    /// characters in identifiers between different [scripts].
    ///
    /// [scripts]: https://en.wikipedia.org/wiki/Script_(Unicode)
    ///
    /// ### Example
    ///
    /// ```rust
    /// // The Japanese katakana character エ can be confused with the Han character 工.
    /// const エ: &'static str = "アイウ";
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// This lint warns when characters between different scripts may appear
    /// visually similar, which can cause confusion.
    ///
    /// If the crate contains other identifiers in the same script that have
    /// non-confusable characters, then this lint will *not* be issued. For
    /// example, if the example given above has another identifier with
    /// katakana characters (such as `let カタカナ = 123;`), then this indicates
    /// that you are intentionally using katakana, and it will not warn about
    /// it.
    ///
    /// Note that the set of confusable characters may change over time.
    /// Beware that if you "forbid" this lint that existing code may fail in
    /// the future.
    pub MIXED_SCRIPT_CONFUSABLES,
    Warn,
    "detects Unicode scripts whose mixed script confusables codepoints are solely used",
    crate_level_only
}

declare_lint_pass!(NonAsciiIdents => [NON_ASCII_IDENTS, UNCOMMON_CODEPOINTS, CONFUSABLE_IDENTS, MIXED_SCRIPT_CONFUSABLES]);

impl EarlyLintPass for NonAsciiIdents {
    fn check_crate(&mut self, cx: &EarlyContext<'_>, _: &ast::Crate) {
        use rustc_session::lint::Level;
        use rustc_span::Span;
        use std::collections::BTreeMap;
        use unicode_security::GeneralSecurityProfile;

        let check_non_ascii_idents = cx.builder.lint_level(NON_ASCII_IDENTS).0 != Level::Allow;
        let check_uncommon_codepoints =
            cx.builder.lint_level(UNCOMMON_CODEPOINTS).0 != Level::Allow;
        let check_confusable_idents = cx.builder.lint_level(CONFUSABLE_IDENTS).0 != Level::Allow;
        let check_mixed_script_confusables =
            cx.builder.lint_level(MIXED_SCRIPT_CONFUSABLES).0 != Level::Allow;

        if !check_non_ascii_idents
            && !check_uncommon_codepoints
            && !check_confusable_idents
            && !check_mixed_script_confusables
        {
            return;
        }

        let mut has_non_ascii_idents = false;
        let symbols = cx.sess.parse_sess.symbol_gallery.symbols.lock();

        // Sort by `Span` so that error messages make sense with respect to the
        // order of identifier locations in the code.
        let mut symbols: Vec<_> = symbols.iter().collect();
        symbols.sort_by_key(|k| k.1);

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
            let mut skeleton_map: FxHashMap<Symbol, (Symbol, Span, bool)> =
                FxHashMap::with_capacity_and_hasher(symbols.len(), Default::default());
            let mut skeleton_buf = String::new();

            for (&symbol, &sp) in symbols.iter() {
                use unicode_security::confusable_detection::skeleton;

                let symbol_str = symbol.as_str();
                let is_ascii = symbol_str.is_ascii();

                // Get the skeleton as a `Symbol`.
                skeleton_buf.clear();
                skeleton_buf.extend(skeleton(&symbol_str));
                let skeleton_sym = if *symbol_str == *skeleton_buf {
                    symbol
                } else {
                    Symbol::intern(&skeleton_buf)
                };

                skeleton_map
                    .entry(skeleton_sym)
                    .and_modify(|(existing_symbol, existing_span, existing_is_ascii)| {
                        if !*existing_is_ascii || !is_ascii {
                            cx.struct_span_lint(CONFUSABLE_IDENTS, sp, |lint| {
                                lint.build(&format!(
                                    "identifier pair considered confusable between `{}` and `{}`",
                                    existing_symbol.as_str(),
                                    symbol.as_str()
                                ))
                                .span_label(
                                    *existing_span,
                                    "this is where the previous identifier occurred",
                                )
                                .emit();
                            });
                        }
                        if *existing_is_ascii && !is_ascii {
                            *existing_symbol = symbol;
                            *existing_span = sp;
                            *existing_is_ascii = is_ascii;
                        }
                    })
                    .or_insert((symbol, sp, is_ascii));
            }
        }

        if has_non_ascii_idents && check_mixed_script_confusables {
            use unicode_security::is_potential_mixed_script_confusable_char;
            use unicode_security::mixed_script::AugmentedScriptSet;

            #[derive(Clone)]
            enum ScriptSetUsage {
                Suspicious(Vec<char>, Span),
                Verified,
            }

            let mut script_states: FxHashMap<AugmentedScriptSet, ScriptSetUsage> =
                FxHashMap::default();
            let latin_augmented_script_set = AugmentedScriptSet::for_char('A');
            script_states.insert(latin_augmented_script_set, ScriptSetUsage::Verified);

            let mut has_suspicous = false;
            for (symbol, &sp) in symbols.iter() {
                let symbol_str = symbol.as_str();
                for ch in symbol_str.chars() {
                    if ch.is_ascii() {
                        // all ascii characters are covered by exception.
                        continue;
                    }
                    if !GeneralSecurityProfile::identifier_allowed(ch) {
                        // this character is covered by `uncommon_codepoints` lint.
                        continue;
                    }
                    let augmented_script_set = AugmentedScriptSet::for_char(ch);
                    script_states
                        .entry(augmented_script_set)
                        .and_modify(|existing_state| {
                            if let ScriptSetUsage::Suspicious(ch_list, _) = existing_state {
                                if is_potential_mixed_script_confusable_char(ch) {
                                    ch_list.push(ch);
                                } else {
                                    *existing_state = ScriptSetUsage::Verified;
                                }
                            }
                        })
                        .or_insert_with(|| {
                            if !is_potential_mixed_script_confusable_char(ch) {
                                ScriptSetUsage::Verified
                            } else {
                                has_suspicous = true;
                                ScriptSetUsage::Suspicious(vec![ch], sp)
                            }
                        });
                }
            }

            if has_suspicous {
                let verified_augmented_script_sets = script_states
                    .iter()
                    .flat_map(|(k, v)| match v {
                        ScriptSetUsage::Verified => Some(*k),
                        _ => None,
                    })
                    .collect::<Vec<_>>();

                // we're sorting the output here.
                let mut lint_reports: BTreeMap<(Span, Vec<char>), AugmentedScriptSet> =
                    BTreeMap::new();

                'outerloop: for (augment_script_set, usage) in script_states {
                    let (mut ch_list, sp) = match usage {
                        ScriptSetUsage::Verified => continue,
                        ScriptSetUsage::Suspicious(ch_list, sp) => (ch_list, sp),
                    };

                    if augment_script_set.is_all() {
                        continue;
                    }

                    for existing in verified_augmented_script_sets.iter() {
                        if existing.is_all() {
                            continue;
                        }
                        let mut intersect = *existing;
                        intersect.intersect_with(augment_script_set);
                        if !intersect.is_empty() && !intersect.is_all() {
                            continue 'outerloop;
                        }
                    }

                    // We sort primitive chars here and can use unstable sort
                    ch_list.sort_unstable();
                    ch_list.dedup();
                    lint_reports.insert((sp, ch_list), augment_script_set);
                }

                for ((sp, ch_list), script_set) in lint_reports {
                    cx.struct_span_lint(MIXED_SCRIPT_CONFUSABLES, sp, |lint| {
                        let message = format!(
                            "the usage of Script Group `{}` in this crate consists solely of mixed script confusables",
                            script_set);
                        let mut note = "the usage includes ".to_string();
                        for (idx, ch) in ch_list.into_iter().enumerate() {
                            if idx != 0 {
                                note += ", ";
                            }
                            let char_info = format!("'{}' (U+{:04X})", ch, ch as u32);
                            note += &char_info;
                        }
                        lint.build(&message).note(&note).note("please recheck to make sure their usages are indeed what you want").emit()
                    });
                }
            }
        }
    }
}
