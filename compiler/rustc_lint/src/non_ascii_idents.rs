use crate::{EarlyContext, EarlyLintPass, LintContext};
use rustc_ast as ast;
use rustc_data_structures::fx::FxHashMap;
use rustc_span::symbol::Symbol;

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

declare_lint! {
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

                    ch_list.sort();
                    ch_list.dedup();
                    lint_reports.insert((sp, ch_list), augment_script_set);
                }

                for ((sp, ch_list), script_set) in lint_reports {
                    cx.struct_span_lint(MIXED_SCRIPT_CONFUSABLES, sp, |lint| {
                        let message = format!(
                            "The usage of Script Group `{}` in this crate consists solely of mixed script confusables",
                            script_set);
                        let mut note = "The usage includes ".to_string();
                        for (idx, ch) in ch_list.into_iter().enumerate() {
                            if idx != 0 {
                                note += ", ";
                            }
                            let char_info = format!("'{}' (U+{:04X})", ch, ch as u32);
                            note += &char_info;
                        }
                        note += ".";
                        lint.build(&message).note(&note).note("Please recheck to make sure their usages are indeed what you want.").emit()
                    });
                }
            }
        }
    }
}
