use rustc_errors::{Applicability, Diag};
use rustc_session::{config::ExpectedValues, Session};
use rustc_span::edit_distance::find_best_match_for_name;
use rustc_span::{sym, Span, Symbol};

const MAX_CHECK_CFG_NAMES_OR_VALUES: usize = 35;

fn check_cfg_expected_note(
    sess: &Session,
    possibilities: &[Symbol],
    type_: &str,
    name: Option<Symbol>,
    suffix: &str,
) -> String {
    use std::fmt::Write;

    let n_possibilities = if sess.opts.unstable_opts.check_cfg_all_expected {
        possibilities.len()
    } else {
        std::cmp::min(possibilities.len(), MAX_CHECK_CFG_NAMES_OR_VALUES)
    };

    let mut possibilities = possibilities.iter().map(Symbol::as_str).collect::<Vec<_>>();
    possibilities.sort();

    let and_more = possibilities.len().saturating_sub(n_possibilities);
    let possibilities = possibilities[..n_possibilities].join("`, `");

    let mut note = String::with_capacity(50 + possibilities.len());

    write!(&mut note, "expected {type_}").unwrap();
    if let Some(name) = name {
        write!(&mut note, " for `{name}`").unwrap();
    }
    write!(&mut note, " are: {suffix}`{possibilities}`").unwrap();
    if and_more > 0 {
        write!(&mut note, " and {and_more} more").unwrap();
    }

    note
}

pub(super) fn unexpected_cfg_name(
    sess: &Session,
    diag: &mut Diag<'_, ()>,
    (name, name_span): (Symbol, Span),
    value: Option<(Symbol, Span)>,
) {
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
    let mut is_feature_cfg = name == sym::feature;

    if is_feature_cfg && is_from_cargo {
        diag.help("consider defining some features in `Cargo.toml`");
    // Suggest the most probable if we found one
    } else if let Some(best_match) = find_best_match_for_name(&possibilities, name, None) {
        if let Some(ExpectedValues::Some(best_match_values)) =
            sess.psess.check_config.expecteds.get(&best_match)
        {
            // We will soon sort, so the initial order does not matter.
            #[allow(rustc::potential_query_instability)]
            let mut possibilities =
                best_match_values.iter().flatten().map(Symbol::as_str).collect::<Vec<_>>();
            possibilities.sort();

            let mut should_print_possibilities = true;
            if let Some((value, value_span)) = value {
                if best_match_values.contains(&Some(value)) {
                    diag.span_suggestion(
                        name_span,
                        "there is a config with a similar name and value",
                        best_match,
                        Applicability::MaybeIncorrect,
                    );
                    should_print_possibilities = false;
                } else if best_match_values.contains(&None) {
                    diag.span_suggestion(
                        name_span.to(value_span),
                        "there is a config with a similar name and no value",
                        best_match,
                        Applicability::MaybeIncorrect,
                    );
                    should_print_possibilities = false;
                } else if let Some(first_value) = possibilities.first() {
                    diag.span_suggestion(
                        name_span.to(value_span),
                        "there is a config with a similar name and different values",
                        format!("{best_match} = \"{first_value}\""),
                        Applicability::MaybeIncorrect,
                    );
                } else {
                    diag.span_suggestion(
                        name_span.to(value_span),
                        "there is a config with a similar name and different values",
                        best_match,
                        Applicability::MaybeIncorrect,
                    );
                };
            } else {
                diag.span_suggestion(
                    name_span,
                    "there is a config with a similar name",
                    best_match,
                    Applicability::MaybeIncorrect,
                );
            }

            if !possibilities.is_empty() && should_print_possibilities {
                let possibilities = possibilities.join("`, `");
                diag.help(format!("expected values for `{best_match}` are: `{possibilities}`"));
            }
        } else {
            diag.span_suggestion(
                name_span,
                "there is a config with a similar name",
                best_match,
                Applicability::MaybeIncorrect,
            );
        }

        is_feature_cfg |= best_match == sym::feature;
    } else {
        if !names_possibilities.is_empty() && names_possibilities.len() <= 3 {
            names_possibilities.sort();
            for cfg_name in names_possibilities.iter() {
                diag.span_suggestion(
                    name_span,
                    "found config with similar value",
                    format!("{cfg_name} = \"{name}\""),
                    Applicability::MaybeIncorrect,
                );
            }
        }
        if !possibilities.is_empty() {
            diag.help_once(check_cfg_expected_note(sess, &possibilities, "names", None, ""));
        }
    }

    let inst = if let Some((value, _value_span)) = value {
        let pre = if is_from_cargo { "\\" } else { "" };
        format!("cfg({name}, values({pre}\"{value}{pre}\"))")
    } else {
        format!("cfg({name})")
    };

    if is_from_cargo {
        if !is_feature_cfg {
            diag.help(format!("consider using a Cargo feature instead or adding `println!(\"cargo:rustc-check-cfg={inst}\");` to the top of a `build.rs`"));
        }
        diag.note("see <https://doc.rust-lang.org/nightly/cargo/reference/unstable.html#check-cfg> for more information about checking conditional configuration");
    } else {
        diag.help(format!("to expect this configuration use `--check-cfg={inst}`"));
        diag.note("see <https://doc.rust-lang.org/nightly/unstable-book/compiler-flags/check-cfg.html> for more information about checking conditional configuration");
    }
}

pub(super) fn unexpected_cfg_value(
    sess: &Session,
    diag: &mut Diag<'_, ()>,
    (name, name_span): (Symbol, Span),
    value: Option<(Symbol, Span)>,
) {
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

    // Show the full list if all possible values for a given name, but don't do it
    // for names as the possibilities could be very long
    if !possibilities.is_empty() {
        diag.note(check_cfg_expected_note(
            sess,
            &possibilities,
            "values",
            Some(name),
            if have_none_possibility { "(none), " } else { "" },
        ));

        if let Some((value, value_span)) = value {
            // Suggest the most probable if we found one
            if let Some(best_match) = find_best_match_for_name(&possibilities, value, None) {
                diag.span_suggestion(
                    value_span,
                    "there is a expected value with a similar name",
                    format!("\"{best_match}\""),
                    Applicability::MaybeIncorrect,
                );
            }
        } else if let &[first_possibility] = &possibilities[..] {
            diag.span_suggestion(
                name_span.shrink_to_hi(),
                "specify a config value",
                format!(" = \"{first_possibility}\""),
                Applicability::MaybeIncorrect,
            );
        }
    } else if have_none_possibility {
        diag.note(format!("no expected value for `{name}`"));
        if let Some((_value, value_span)) = value {
            diag.span_suggestion(
                name_span.shrink_to_hi().to(value_span),
                "remove the value",
                "",
                Applicability::MaybeIncorrect,
            );
        }
    } else {
        diag.note(format!("no expected values for `{name}`"));

        let sp = if let Some((_value, value_span)) = value {
            name_span.to(value_span)
        } else {
            name_span
        };
        diag.span_suggestion(sp, "remove the condition", "", Applicability::MaybeIncorrect);
    }

    // We don't want to suggest adding values to well known names
    // since those are defined by rustc it-self. Users can still
    // do it if they want, but should not encourage them.
    let is_cfg_a_well_know_name = sess.psess.check_config.well_known_names.contains(&name);

    let inst = if let Some((value, _value_span)) = value {
        let pre = if is_from_cargo { "\\" } else { "" };
        format!("cfg({name}, values({pre}\"{value}{pre}\"))")
    } else {
        format!("cfg({name})")
    };

    if is_from_cargo {
        if name == sym::feature {
            if let Some((value, _value_span)) = value {
                diag.help(format!("consider adding `{value}` as a feature in `Cargo.toml`"));
            } else {
                diag.help("consider defining some features in `Cargo.toml`");
            }
        } else if !is_cfg_a_well_know_name {
            diag.help(format!("consider using a Cargo feature instead or adding `println!(\"cargo:rustc-check-cfg={inst}\");` to the top of a `build.rs`"));
        }
        diag.note("see <https://doc.rust-lang.org/nightly/cargo/reference/unstable.html#check-cfg> for more information about checking conditional configuration");
    } else {
        if !is_cfg_a_well_know_name {
            diag.help(format!("to expect this configuration use `--check-cfg={inst}`"));
        }
        diag.note("see <https://doc.rust-lang.org/nightly/unstable-book/compiler-flags/check-cfg.html> for more information about checking conditional configuration");
    }
}
