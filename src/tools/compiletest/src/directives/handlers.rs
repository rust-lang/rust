use std::collections::HashMap;
use std::sync::{Arc, LazyLock};

use crate::common::Config;
use crate::directives::{
    DirectiveLine, NormalizeKind, NormalizeRule, TestProps, parse_and_update_aux,
    parse_edition_range, split_flags,
};
use crate::errors::ErrorKind;

pub(crate) static DIRECTIVE_HANDLERS_MAP: LazyLock<HashMap<&str, Handler>> =
    LazyLock::new(make_directive_handlers_map);

#[derive(Clone)]
pub(crate) struct Handler {
    handler_fn: Arc<dyn Fn(HandlerArgs<'_>) + Send + Sync>,
}

impl Handler {
    pub(crate) fn handle(&self, config: &Config, line: &DirectiveLine<'_>, props: &mut TestProps) {
        (self.handler_fn)(HandlerArgs { config, line, props })
    }
}

struct HandlerArgs<'a> {
    config: &'a Config,
    line: &'a DirectiveLine<'a>,
    props: &'a mut TestProps,
}

/// Intermediate data structure, used for defining a list of handlers.
struct NamedHandler {
    names: Vec<&'static str>,
    handler: Handler,
}

/// Helper function to create a simple handler, so that changes can be made
/// to the handler struct without disturbing existing handler declarations.
fn handler(
    name: &'static str,
    handler_fn: impl Fn(&Config, &DirectiveLine<'_>, &mut TestProps) + Send + Sync + 'static,
) -> NamedHandler {
    multi_handler(&[name], handler_fn)
}

/// Associates the same handler function with multiple directive names.
fn multi_handler(
    names: &[&'static str],
    handler_fn: impl Fn(&Config, &DirectiveLine<'_>, &mut TestProps) + Send + Sync + 'static,
) -> NamedHandler {
    NamedHandler {
        names: names.to_owned(),
        handler: Handler {
            handler_fn: Arc::new(move |args| handler_fn(args.config, args.line, args.props)),
        },
    }
}

fn make_directive_handlers_map() -> HashMap<&'static str, Handler> {
    use crate::directives::directives::*;

    // FIXME(Zalathar): Now that most directive-processing has been extracted
    // into individual handlers, there should be many opportunities to simplify
    // these handlers, e.g. by getting rid of now-redundant name checks.

    let handlers: Vec<NamedHandler> = vec![
        handler(ERROR_PATTERN, |config, ln, props| {
            config.push_name_value_directive(ln, ERROR_PATTERN, &mut props.error_patterns, |r| r);
        }),
        handler(REGEX_ERROR_PATTERN, |config, ln, props| {
            config.push_name_value_directive(
                ln,
                REGEX_ERROR_PATTERN,
                &mut props.regex_error_patterns,
                |r| r,
            );
        }),
        handler(DOC_FLAGS, |config, ln, props| {
            config.push_name_value_directive(ln, DOC_FLAGS, &mut props.doc_flags, |r| r);
        }),
        handler(COMPILE_FLAGS, |config, ln, props| {
            if let Some(flags) = config.parse_name_value_directive(ln, COMPILE_FLAGS) {
                let flags = split_flags(&flags);
                // FIXME(#147955): Extract and unify this with other handlers that
                // check compiler flags, e.g. MINICORE_COMPILE_FLAGS.
                for (i, flag) in flags.iter().enumerate() {
                    if flag == "--edition" || flag.starts_with("--edition=") {
                        panic!("you must use `//@ edition` to configure the edition");
                    }
                    if (flag == "-C"
                        && flags.get(i + 1).is_some_and(|v| v.starts_with("incremental=")))
                        || flag.starts_with("-Cincremental=")
                    {
                        panic!("you must use `//@ incremental` to enable incremental compilation");
                    }
                }
                props.compile_flags.extend(flags);
            }
        }),
        handler("edition", |config, ln, props| {
            if let Some(range) = parse_edition_range(config, ln) {
                props.edition = Some(range.edition_to_test(config.edition));
            }
        }),
        handler("revisions", |config, ln, props| {
            config.parse_and_update_revisions(ln, &mut props.revisions);
        }),
        handler(RUN_FLAGS, |config, ln, props| {
            if let Some(flags) = config.parse_name_value_directive(ln, RUN_FLAGS) {
                props.run_flags.extend(split_flags(&flags));
            }
        }),
        handler("pp-exact", |config, ln, props| {
            if props.pp_exact.is_none() {
                props.pp_exact = config.parse_pp_exact(ln);
            }
        }),
        handler(SHOULD_ICE, |config, ln, props| {
            config.set_name_directive(ln, SHOULD_ICE, &mut props.should_ice);
        }),
        handler(BUILD_AUX_DOCS, |config, ln, props| {
            config.set_name_directive(ln, BUILD_AUX_DOCS, &mut props.build_aux_docs);
        }),
        handler(UNIQUE_DOC_OUT_DIR, |config, ln, props| {
            config.set_name_directive(ln, UNIQUE_DOC_OUT_DIR, &mut props.unique_doc_out_dir);
        }),
        handler(FORCE_HOST, |config, ln, props| {
            config.set_name_directive(ln, FORCE_HOST, &mut props.force_host);
        }),
        handler(CHECK_STDOUT, |config, ln, props| {
            config.set_name_directive(ln, CHECK_STDOUT, &mut props.check_stdout);
        }),
        handler(CHECK_RUN_RESULTS, |config, ln, props| {
            config.set_name_directive(ln, CHECK_RUN_RESULTS, &mut props.check_run_results);
        }),
        handler(DONT_CHECK_COMPILER_STDOUT, |config, ln, props| {
            config.set_name_directive(
                ln,
                DONT_CHECK_COMPILER_STDOUT,
                &mut props.dont_check_compiler_stdout,
            );
        }),
        handler(DONT_CHECK_COMPILER_STDERR, |config, ln, props| {
            config.set_name_directive(
                ln,
                DONT_CHECK_COMPILER_STDERR,
                &mut props.dont_check_compiler_stderr,
            );
        }),
        handler(NO_PREFER_DYNAMIC, |config, ln, props| {
            config.set_name_directive(ln, NO_PREFER_DYNAMIC, &mut props.no_prefer_dynamic);
        }),
        handler(PRETTY_MODE, |config, ln, props| {
            if let Some(m) = config.parse_name_value_directive(ln, PRETTY_MODE) {
                props.pretty_mode = m;
            }
        }),
        handler(PRETTY_COMPARE_ONLY, |config, ln, props| {
            config.set_name_directive(ln, PRETTY_COMPARE_ONLY, &mut props.pretty_compare_only);
        }),
        multi_handler(
            &[AUX_BUILD, AUX_BIN, AUX_CRATE, PROC_MACRO, AUX_CODEGEN_BACKEND],
            |config, ln, props| {
                // Call a helper method to deal with aux-related directives.
                parse_and_update_aux(config, ln, &mut props.aux);
            },
        ),
        handler(EXEC_ENV, |config, ln, props| {
            config.push_name_value_directive(ln, EXEC_ENV, &mut props.exec_env, Config::parse_env);
        }),
        handler(UNSET_EXEC_ENV, |config, ln, props| {
            config.push_name_value_directive(ln, UNSET_EXEC_ENV, &mut props.unset_exec_env, |r| {
                r.trim().to_owned()
            });
        }),
        handler(RUSTC_ENV, |config, ln, props| {
            config.push_name_value_directive(
                ln,
                RUSTC_ENV,
                &mut props.rustc_env,
                Config::parse_env,
            );
        }),
        handler(UNSET_RUSTC_ENV, |config, ln, props| {
            config.push_name_value_directive(
                ln,
                UNSET_RUSTC_ENV,
                &mut props.unset_rustc_env,
                |r| r.trim().to_owned(),
            );
        }),
        handler(FORBID_OUTPUT, |config, ln, props| {
            config.push_name_value_directive(ln, FORBID_OUTPUT, &mut props.forbid_output, |r| r);
        }),
        handler(CHECK_TEST_LINE_NUMBERS_MATCH, |config, ln, props| {
            config.set_name_directive(
                ln,
                CHECK_TEST_LINE_NUMBERS_MATCH,
                &mut props.check_test_line_numbers_match,
            );
        }),
        multi_handler(&["check-pass", "build-pass", "run-pass"], |config, ln, props| {
            props.update_pass_mode(ln, config);
        }),
        multi_handler(
            &["check-fail", "build-fail", "run-fail", "run-crash", "run-fail-or-crash"],
            |config, ln, props| {
                props.update_fail_mode(ln, config);
            },
        ),
        handler(IGNORE_PASS, |config, ln, props| {
            config.set_name_directive(ln, IGNORE_PASS, &mut props.ignore_pass);
        }),
        multi_handler(
            &[
                "normalize-stdout",
                "normalize-stderr",
                "normalize-stderr-32bit",
                "normalize-stderr-64bit",
            ],
            |config, ln, props| {
                if let Some(NormalizeRule { kind, regex, replacement }) =
                    config.parse_custom_normalization(ln)
                {
                    let rule_tuple = (regex, replacement);
                    match kind {
                        NormalizeKind::Stdout => props.normalize_stdout.push(rule_tuple),
                        NormalizeKind::Stderr => props.normalize_stderr.push(rule_tuple),
                        NormalizeKind::Stderr32bit => {
                            if config.target_cfg().pointer_width == 32 {
                                props.normalize_stderr.push(rule_tuple);
                            }
                        }
                        NormalizeKind::Stderr64bit => {
                            if config.target_cfg().pointer_width == 64 {
                                props.normalize_stderr.push(rule_tuple);
                            }
                        }
                    }
                }
            },
        ),
        handler(FAILURE_STATUS, |config, ln, props| {
            if let Some(code) = config
                .parse_name_value_directive(ln, FAILURE_STATUS)
                .and_then(|code| code.trim().parse::<i32>().ok())
            {
                props.failure_status = Some(code);
            }
        }),
        handler(DONT_CHECK_FAILURE_STATUS, |config, ln, props| {
            config.set_name_directive(
                ln,
                DONT_CHECK_FAILURE_STATUS,
                &mut props.dont_check_failure_status,
            );
        }),
        handler(RUN_RUSTFIX, |config, ln, props| {
            config.set_name_directive(ln, RUN_RUSTFIX, &mut props.run_rustfix);
        }),
        handler(RUSTFIX_ONLY_MACHINE_APPLICABLE, |config, ln, props| {
            config.set_name_directive(
                ln,
                RUSTFIX_ONLY_MACHINE_APPLICABLE,
                &mut props.rustfix_only_machine_applicable,
            );
        }),
        handler(ASSEMBLY_OUTPUT, |config, ln, props| {
            config.set_name_value_directive(ln, ASSEMBLY_OUTPUT, &mut props.assembly_output, |r| {
                r.trim().to_string()
            });
        }),
        handler(STDERR_PER_BITWIDTH, |config, ln, props| {
            config.set_name_directive(ln, STDERR_PER_BITWIDTH, &mut props.stderr_per_bitwidth);
        }),
        handler(INCREMENTAL, |config, ln, props| {
            config.set_name_directive(ln, INCREMENTAL, &mut props.incremental);
        }),
        handler(KNOWN_BUG, |config, ln, props| {
            // Unlike the other `name_value_directive`s this needs to be handled manually,
            // because it sets a `bool` flag.
            if let Some(known_bug) = config.parse_name_value_directive(ln, KNOWN_BUG) {
                let known_bug = known_bug.trim();
                if known_bug == "unknown"
                    || known_bug.split(',').all(|issue_ref| {
                        issue_ref
                            .trim()
                            .split_once('#')
                            .filter(|(_, number)| number.chars().all(|digit| digit.is_numeric()))
                            .is_some()
                    })
                {
                    props.known_bug = true;
                } else {
                    panic!(
                        "Invalid known-bug value: {known_bug}\nIt requires comma-separated issue references (`#000` or `chalk#000`) or `known-bug: unknown`."
                    );
                }
            } else if config.parse_name_directive(ln, KNOWN_BUG) {
                panic!(
                    "Invalid known-bug attribute, requires comma-separated issue references (`#000` or `chalk#000`) or `known-bug: unknown`."
                );
            }
        }),
        handler(TEST_MIR_PASS, |config, ln, props| {
            config.set_name_value_directive(ln, TEST_MIR_PASS, &mut props.mir_unit_test, |s| {
                s.trim().to_string()
            });
        }),
        handler(REMAP_SRC_BASE, |config, ln, props| {
            config.set_name_directive(ln, REMAP_SRC_BASE, &mut props.remap_src_base);
        }),
        handler(LLVM_COV_FLAGS, |config, ln, props| {
            if let Some(flags) = config.parse_name_value_directive(ln, LLVM_COV_FLAGS) {
                props.llvm_cov_flags.extend(split_flags(&flags));
            }
        }),
        handler(FILECHECK_FLAGS, |config, ln, props| {
            if let Some(flags) = config.parse_name_value_directive(ln, FILECHECK_FLAGS) {
                props.filecheck_flags.extend(split_flags(&flags));
            }
        }),
        handler(NO_AUTO_CHECK_CFG, |config, ln, props| {
            config.set_name_directive(ln, NO_AUTO_CHECK_CFG, &mut props.no_auto_check_cfg);
        }),
        handler(ADD_MINICORE, |config, ln, props| {
            props.update_add_minicore(ln, config);
        }),
        handler(MINICORE_COMPILE_FLAGS, |config, ln, props| {
            if let Some(flags) = config.parse_name_value_directive(ln, MINICORE_COMPILE_FLAGS) {
                let flags = split_flags(&flags);
                // FIXME(#147955): Extract and unify this with other handlers that
                // check compiler flags, e.g. COMPILE_FLAGS.
                for flag in &flags {
                    if flag == "--edition" || flag.starts_with("--edition=") {
                        panic!("you must use `//@ edition` to configure the edition");
                    }
                }
                props.minicore_compile_flags.extend(flags);
            }
        }),
        handler(DONT_REQUIRE_ANNOTATIONS, |config, ln, props| {
            if let Some(err_kind) = config.parse_name_value_directive(ln, DONT_REQUIRE_ANNOTATIONS)
            {
                props
                    .dont_require_annotations
                    .insert(ErrorKind::expect_from_user_str(err_kind.trim()));
            }
        }),
        handler(DISABLE_GDB_PRETTY_PRINTERS, |config, ln, props| {
            config.set_name_directive(
                ln,
                DISABLE_GDB_PRETTY_PRINTERS,
                &mut props.disable_gdb_pretty_printers,
            );
        }),
        handler(COMPARE_OUTPUT_BY_LINES, |config, ln, props| {
            config.set_name_directive(
                ln,
                COMPARE_OUTPUT_BY_LINES,
                &mut props.compare_output_by_lines,
            );
        }),
    ];

    handlers
        .into_iter()
        .flat_map(|NamedHandler { names, handler }| {
            names.into_iter().map(move |name| (name, Handler::clone(&handler)))
        })
        .collect()
}
