use rustc_hir::def_id::LOCAL_CRATE;
use rustc_middle::bug;
use rustc_middle::ty::TyCtxt;
use rustc_session::Session;
use rustc_session::config::ExpectedValues;
use rustc_span::edit_distance::find_best_match_for_name;
use rustc_span::{ExpnKind, Ident, Span, Symbol, sym};

use crate::lints;

const MAX_CHECK_CFG_NAMES_OR_VALUES: usize = 35;

enum FilterWellKnownNames {
    Yes,
    No,
}

fn sort_and_truncate_possibilities(
    sess: &Session,
    mut possibilities: Vec<Symbol>,
    filter_well_known_names: FilterWellKnownNames,
) -> (Vec<Symbol>, usize) {
    let possibilities_len = possibilities.len();

    let n_possibilities = if sess.opts.unstable_opts.check_cfg_all_expected {
        possibilities.len()
    } else {
        match filter_well_known_names {
            FilterWellKnownNames::Yes => {
                possibilities.retain(|cfg_name| {
                    !sess.psess.check_config.well_known_names.contains(cfg_name)
                });
            }
            FilterWellKnownNames::No => {}
        };
        std::cmp::min(possibilities.len(), MAX_CHECK_CFG_NAMES_OR_VALUES)
    };

    possibilities.sort_by(|s1, s2| s1.as_str().cmp(s2.as_str()));

    let and_more = possibilities_len.saturating_sub(n_possibilities);
    possibilities.truncate(n_possibilities);
    (possibilities, and_more)
}

enum EscapeQuotes {
    Yes,
    No,
}

fn to_check_cfg_arg(name: Ident, value: Option<Symbol>, quotes: EscapeQuotes) -> String {
    if let Some(value) = value {
        let value = str::escape_debug(value.as_str()).to_string();
        let values = match quotes {
            EscapeQuotes::Yes => format!("\\\"{}\\\"", value.replace("\"", "\\\\\\\\\"")),
            EscapeQuotes::No => format!("\"{value}\""),
        };
        format!("cfg({name}, values({values}))")
    } else {
        format!("cfg({name})")
    }
}

fn cargo_help_sub(
    sess: &Session,
    inst: &impl Fn(EscapeQuotes) -> String,
) -> lints::UnexpectedCfgCargoHelp {
    // We don't want to suggest the `build.rs` way to expected cfgs if we are already in a
    // `build.rs`. We therefor do a best effort check (looking if the `--crate-name` is
    // `build_script_build`) to try to figure out if we are building a Cargo build script

    let unescaped = &inst(EscapeQuotes::No);
    if matches!(&sess.opts.crate_name, Some(crate_name) if crate_name == "build_script_build") {
        lints::UnexpectedCfgCargoHelp::lint_cfg(unescaped)
    } else {
        lints::UnexpectedCfgCargoHelp::lint_cfg_and_build_rs(unescaped, &inst(EscapeQuotes::Yes))
    }
}

fn rustc_macro_help(span: Span) -> Option<lints::UnexpectedCfgRustcMacroHelp> {
    let oexpn = span.ctxt().outer_expn_data();
    if let Some(def_id) = oexpn.macro_def_id
        && let ExpnKind::Macro(macro_kind, macro_name) = oexpn.kind
        && def_id.krate != LOCAL_CRATE
    {
        Some(lints::UnexpectedCfgRustcMacroHelp { macro_kind: macro_kind.descr(), macro_name })
    } else {
        None
    }
}

fn cargo_macro_help(
    tcx: Option<TyCtxt<'_>>,
    span: Span,
) -> Option<lints::UnexpectedCfgCargoMacroHelp> {
    let oexpn = span.ctxt().outer_expn_data();
    if let Some(def_id) = oexpn.macro_def_id
        && let ExpnKind::Macro(macro_kind, macro_name) = oexpn.kind
        && def_id.krate != LOCAL_CRATE
        && let Some(tcx) = tcx
    {
        Some(lints::UnexpectedCfgCargoMacroHelp {
            macro_kind: macro_kind.descr(),
            macro_name,
            crate_name: tcx.crate_name(def_id.krate),
        })
    } else {
        None
    }
}

pub(super) fn unexpected_cfg_name(
    sess: &Session,
    tcx: Option<TyCtxt<'_>>,
    (name, name_span): (Symbol, Span),
    value: Option<(Symbol, Span)>,
) -> lints::UnexpectedCfgName {
    #[allow(rustc::potential_query_instability)]
    let possibilities: Vec<Symbol> = sess.psess.check_config.expecteds.keys().copied().collect();

    let mut names_possibilities: Vec<_> = if value.is_none() {
        // We later sort and display all the possibilities, so the order here does not matter.
        #[allow(rustc::potential_query_instability)]
        sess.psess
            .check_config
            .expecteds
            .iter()
            .filter_map(|(k, v)| match v {
                ExpectedValues::Some(v) if v.contains(&Some(name)) => Some(k),
                _ => None,
            })
            .collect()
    } else {
        Vec::new()
    };

    let is_from_cargo = rustc_session::utils::was_invoked_from_cargo();
    let is_from_external_macro = name_span.in_external_macro(sess.source_map());
    let mut is_feature_cfg = name == sym::feature;

    let code_sugg = if is_feature_cfg && is_from_cargo {
        lints::unexpected_cfg_name::CodeSuggestion::DefineFeatures
    // Suggest correct `version("..")` predicate syntax
    } else if let Some((_value, value_span)) = value
        && name == sym::version
    {
        lints::unexpected_cfg_name::CodeSuggestion::VersionSyntax {
            between_name_and_value: name_span.between(value_span),
            after_value: value_span.shrink_to_hi(),
        }
    // Suggest the most probable if we found one
    } else if let Some(best_match) = find_best_match_for_name(&possibilities, name, None) {
        is_feature_cfg |= best_match == sym::feature;

        if let Some(ExpectedValues::Some(best_match_values)) =
            sess.psess.check_config.expecteds.get(&best_match)
        {
            // We will soon sort, so the initial order does not matter.
            #[allow(rustc::potential_query_instability)]
            let mut possibilities = best_match_values.iter().flatten().collect::<Vec<_>>();
            possibilities.sort_by_key(|s| s.as_str());

            let get_possibilities_sub = || {
                if !possibilities.is_empty() {
                    let possibilities =
                        possibilities.iter().copied().cloned().collect::<Vec<_>>().into();
                    Some(lints::unexpected_cfg_name::ExpectedValues { best_match, possibilities })
                } else {
                    None
                }
            };

            let best_match = Ident::new(best_match, name_span);
            if let Some((value, value_span)) = value {
                if best_match_values.contains(&Some(value)) {
                    lints::unexpected_cfg_name::CodeSuggestion::SimilarNameAndValue {
                        span: name_span,
                        code: best_match.to_string(),
                    }
                } else if best_match_values.contains(&None) {
                    lints::unexpected_cfg_name::CodeSuggestion::SimilarNameNoValue {
                        span: name_span.to(value_span),
                        code: best_match.to_string(),
                    }
                } else if let Some(first_value) = possibilities.first() {
                    lints::unexpected_cfg_name::CodeSuggestion::SimilarNameDifferentValues {
                        span: name_span.to(value_span),
                        code: format!("{best_match} = \"{first_value}\""),
                        expected: get_possibilities_sub(),
                    }
                } else {
                    lints::unexpected_cfg_name::CodeSuggestion::SimilarNameDifferentValues {
                        span: name_span.to(value_span),
                        code: best_match.to_string(),
                        expected: get_possibilities_sub(),
                    }
                }
            } else {
                lints::unexpected_cfg_name::CodeSuggestion::SimilarName {
                    span: name_span,
                    code: best_match.to_string(),
                    expected: get_possibilities_sub(),
                }
            }
        } else {
            lints::unexpected_cfg_name::CodeSuggestion::SimilarName {
                span: name_span,
                code: best_match.to_string(),
                expected: None,
            }
        }
    } else {
        let similar_values = if !names_possibilities.is_empty() && names_possibilities.len() <= 3 {
            names_possibilities.sort();
            names_possibilities
                .iter()
                .map(|cfg_name| lints::unexpected_cfg_name::FoundWithSimilarValue {
                    span: name_span,
                    code: format!("{cfg_name} = \"{name}\""),
                })
                .collect()
        } else {
            vec![]
        };

        let (possibilities, and_more) =
            sort_and_truncate_possibilities(sess, possibilities, FilterWellKnownNames::Yes);
        let expected_names = if !possibilities.is_empty() {
            let possibilities: Vec<_> =
                possibilities.into_iter().map(|s| Ident::new(s, name_span)).collect();
            Some(lints::unexpected_cfg_name::ExpectedNames {
                possibilities: possibilities.into(),
                and_more,
            })
        } else {
            None
        };
        lints::unexpected_cfg_name::CodeSuggestion::SimilarValues {
            with_similar_values: similar_values,
            expected_names,
        }
    };

    let inst = |escape_quotes| {
        to_check_cfg_arg(Ident::new(name, name_span), value.map(|(v, _s)| v), escape_quotes)
    };

    let invocation_help = if is_from_cargo {
        let help = if !is_feature_cfg && !is_from_external_macro {
            Some(cargo_help_sub(sess, &inst))
        } else {
            None
        };
        lints::unexpected_cfg_name::InvocationHelp::Cargo {
            help,
            macro_help: cargo_macro_help(tcx, name_span),
        }
    } else {
        let help = lints::UnexpectedCfgRustcHelp::new(&inst(EscapeQuotes::No));
        lints::unexpected_cfg_name::InvocationHelp::Rustc {
            help,
            macro_help: rustc_macro_help(name_span),
        }
    };

    lints::UnexpectedCfgName { code_sugg, invocation_help, name }
}

pub(super) fn unexpected_cfg_value(
    sess: &Session,
    tcx: Option<TyCtxt<'_>>,
    (name, name_span): (Symbol, Span),
    value: Option<(Symbol, Span)>,
) -> lints::UnexpectedCfgValue {
    let Some(ExpectedValues::Some(values)) = &sess.psess.check_config.expecteds.get(&name) else {
        bug!(
            "it shouldn't be possible to have a diagnostic on a value whose name is not in values"
        );
    };
    let mut have_none_possibility = false;
    // We later sort possibilities if it is not empty, so the
    // order here does not matter.
    #[allow(rustc::potential_query_instability)]
    let possibilities: Vec<Symbol> = values
        .iter()
        .inspect(|a| have_none_possibility |= a.is_none())
        .copied()
        .flatten()
        .collect();

    let is_from_cargo = rustc_session::utils::was_invoked_from_cargo();
    let is_from_external_macro = name_span.in_external_macro(sess.source_map());

    // Show the full list if all possible values for a given name, but don't do it
    // for names as the possibilities could be very long
    let code_sugg = if !possibilities.is_empty() {
        let expected_values = {
            let (possibilities, and_more) = sort_and_truncate_possibilities(
                sess,
                possibilities.clone(),
                FilterWellKnownNames::No,
            );
            lints::unexpected_cfg_value::ExpectedValues {
                name,
                have_none_possibility,
                possibilities: possibilities.into(),
                and_more,
            }
        };

        let suggestion = if let Some((value, value_span)) = value {
            // Suggest the most probable if we found one
            if let Some(best_match) = find_best_match_for_name(&possibilities, value, None) {
                Some(lints::unexpected_cfg_value::ChangeValueSuggestion::SimilarName {
                    span: value_span,
                    best_match,
                })
            } else {
                None
            }
        } else if let &[first_possibility] = &possibilities[..] {
            Some(lints::unexpected_cfg_value::ChangeValueSuggestion::SpecifyValue {
                span: name_span.shrink_to_hi(),
                first_possibility,
            })
        } else {
            None
        };

        lints::unexpected_cfg_value::CodeSuggestion::ChangeValue { expected_values, suggestion }
    } else if have_none_possibility {
        let suggestion =
            value.map(|(_value, value_span)| lints::unexpected_cfg_value::RemoveValueSuggestion {
                span: name_span.shrink_to_hi().to(value_span),
            });
        lints::unexpected_cfg_value::CodeSuggestion::RemoveValue { suggestion, name }
    } else {
        let span = if let Some((_value, value_span)) = value {
            name_span.to(value_span)
        } else {
            name_span
        };
        let suggestion = lints::unexpected_cfg_value::RemoveConditionSuggestion { span };
        lints::unexpected_cfg_value::CodeSuggestion::RemoveCondition { suggestion, name }
    };

    // We don't want to encourage people to add values to a well-known names, as these are
    // defined by rustc/Rust itself. Users can still do this if they wish, but should not be
    // encouraged to do so.
    let can_suggest_adding_value = !sess.psess.check_config.well_known_names.contains(&name)
        // Except when working on rustc or the standard library itself, in which case we want to
        // suggest adding these cfgs to the "normal" place because of bootstrapping reasons. As a
        // basic heuristic, we use the "cheat" unstable feature enable method and the
        // non-ui-testing enabled option.
        || (matches!(sess.psess.unstable_features, rustc_feature::UnstableFeatures::Cheat)
            && !sess.opts.unstable_opts.ui_testing);

    let inst = |escape_quotes| {
        to_check_cfg_arg(Ident::new(name, name_span), value.map(|(v, _s)| v), escape_quotes)
    };

    let invocation_help = if is_from_cargo {
        let help = if name == sym::feature && !is_from_external_macro {
            if let Some((value, _value_span)) = value {
                Some(lints::unexpected_cfg_value::CargoHelp::AddFeature { value })
            } else {
                Some(lints::unexpected_cfg_value::CargoHelp::DefineFeatures)
            }
        } else if can_suggest_adding_value && !is_from_external_macro {
            Some(lints::unexpected_cfg_value::CargoHelp::Other(cargo_help_sub(sess, &inst)))
        } else {
            None
        };
        lints::unexpected_cfg_value::InvocationHelp::Cargo {
            help,
            macro_help: cargo_macro_help(tcx, name_span),
        }
    } else {
        let help = if can_suggest_adding_value {
            Some(lints::UnexpectedCfgRustcHelp::new(&inst(EscapeQuotes::No)))
        } else {
            None
        };
        lints::unexpected_cfg_value::InvocationHelp::Rustc {
            help,
            macro_help: rustc_macro_help(name_span),
        }
    };

    lints::UnexpectedCfgValue {
        code_sugg,
        invocation_help,
        has_value: value.is_some(),
        value: value.map_or_else(String::new, |(v, _span)| v.to_string()),
    }
}
