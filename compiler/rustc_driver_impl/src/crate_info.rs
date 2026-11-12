use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use rustc_codegen_ssa::traits::CodegenBackend;
use rustc_feature::find_gated_cfg;
use rustc_interface::passes::collect_crate_types;
use rustc_session::Session;
use rustc_session::output::invalid_output_for_target;
use rustc_span::DUMMY_SP;
use rustc_structures::CrateType;
use rustc_target::json::ToJson;
use rustc_target::spec::{Target, TargetTuple};

use crate::{Compilation, parse_crate_attrs, passes};

pub fn print_crate_info(
    codegen_backend: &dyn CodegenBackend,
    sess: &Session,
    parse_attrs: bool,
) -> Compilation {
    use rustc_session::config::PrintKind::*;
    // This import prevents the following code from using the printing macros
    // used by the rest of the module. Within this function, we only write to
    // the output specified by `sess.io.output_file`.
    #[allow(unused_imports)]
    use {
        crate::do_not_use_safe_print as safe_print, crate::do_not_use_safe_print as safe_println,
    };

    // NativeStaticLibs and LinkArgs are special - printed during linking
    // (empty iterator returns true)
    if sess.opts.prints.iter().all(|p| p.kind == NativeStaticLibs || p.kind == LinkArgs) {
        return Compilation::Continue;
    }

    let attrs = if parse_attrs {
        let result = parse_crate_attrs(sess);
        match result {
            Ok(attrs) => Some(attrs),
            Err(parse_error) => {
                parse_error.emit();
                return Compilation::Stop;
            }
        }
    } else {
        None
    };

    for req in &sess.opts.prints {
        let mut crate_info = String::new();
        macro println_info($($arg:tt)*) {
            crate_info.write_fmt(format_args!("{}\n", format_args!($($arg)*))).unwrap()
        }

        match req.kind {
            TargetList => {
                let mut targets = rustc_target::spec::TARGETS.to_vec();
                targets.sort_unstable();
                println_info!("{}", targets.join("\n"));
            }
            HostTuple => println_info!("{}", rustc_session::config::host_tuple()),
            WasmProcMacroTuple => println_info!("{}", sess.wasm_proc_macro_tuple),
            Sysroot => println_info!("{}", sess.opts.sysroot.path().display()),
            TargetLibdir => println_info!("{}", sess.target_tlib_path.dir.display()),
            TargetSpecJson => {
                println_info!("{}", serde_json::to_string_pretty(&sess.target.to_json()).unwrap());
            }
            TargetSpecJsonSchema => {
                let schema = rustc_target::spec::json_schema();
                println_info!("{}", serde_json::to_string_pretty(&schema).unwrap());
            }
            AllTargetSpecsJson => {
                let mut targets = BTreeMap::new();
                for name in rustc_target::spec::TARGETS {
                    let triple = TargetTuple::from_tuple(name);
                    let target = Target::expect_builtin(&triple);
                    targets.insert(name, target.to_json());
                }
                println_info!("{}", serde_json::to_string_pretty(&targets).unwrap());
            }
            FileNames => {
                let Some(attrs) = attrs.as_ref() else {
                    // no crate attributes, print out an error and exit
                    return Compilation::Continue;
                };
                let t_outputs = rustc_interface::util::build_output_filenames(attrs, sess);
                let crate_name = passes::get_crate_name(sess, attrs);
                let crate_types = collect_crate_types(
                    sess,
                    &codegen_backend.supported_crate_types(sess),
                    codegen_backend.name(),
                    attrs,
                    DUMMY_SP,
                );
                for &style in &crate_types {
                    let fname = rustc_session::output::filename_for_input(
                        sess, style, crate_name, &t_outputs,
                    );
                    println_info!("{}", fname.as_path().file_name().unwrap().to_string_lossy());
                }
            }
            CrateName => {
                let Some(attrs) = attrs.as_ref() else {
                    // no crate attributes, print out an error and exit
                    return Compilation::Continue;
                };
                println_info!("{}", passes::get_crate_name(sess, attrs));
            }
            CrateRootLintLevels => {
                let Some(attrs) = attrs.as_ref() else {
                    // no crate attributes, print out an error and exit
                    return Compilation::Continue;
                };
                let crate_name = passes::get_crate_name(sess, attrs);
                let lint_store = crate::unerased_lint_store(sess);
                let features = rustc_expand::config::features(sess, attrs, crate_name);
                let registered_lint_tools = rustc_resolve::registered_lint_tools_ast(sess, attrs);
                let builder = rustc_lint::LintLevelsBuilder::crate_root(
                    sess,
                    &features,
                    true,
                    lint_store,
                    &registered_lint_tools,
                    attrs,
                );
                for lint in lint_store.get_lints() {
                    if let Some(feature_symbol) = lint.feature_gate
                        && !features.enabled(feature_symbol)
                    {
                        // lint is unstable and feature gate isn't active, don't print
                        continue;
                    }
                    let level = builder.lint_level_spec(lint).level();
                    println_info!("{}={}", lint.name_lower(), level.as_str());
                }
            }
            Cfg => {
                let mut cfgs = sess
                    .config
                    .iter()
                    .filter_map(|&(name, value)| {
                        // On stable, exclude unstable flags.
                        if !sess.is_nightly_build() && find_gated_cfg(name).is_some() {
                            return None;
                        }

                        if let Some(value) = value {
                            Some(format!("{name}=\"{value}\""))
                        } else {
                            Some(name.to_string())
                        }
                    })
                    .collect::<Vec<String>>();

                cfgs.sort();
                for cfg in cfgs {
                    println_info!("{cfg}");
                }
            }
            CheckCfg => {
                let mut check_cfgs: Vec<String> = Vec::with_capacity(410);

                // INSTABILITY: We are sorting the output below.
                #[allow(rustc::potential_query_instability)]
                for (name, expected_values) in &sess.check_config.expecteds {
                    use crate::config::ExpectedValues;
                    match expected_values {
                        ExpectedValues::Any => {
                            check_cfgs.push(format!("cfg({name}, values(any()))"))
                        }
                        ExpectedValues::Some(values) => {
                            let mut values: Vec<_> = values
                                .iter()
                                .map(|value| {
                                    if let Some(value) = value {
                                        format!("\"{value}\"")
                                    } else {
                                        "none()".to_string()
                                    }
                                })
                                .collect();

                            values.sort_unstable();

                            let values = values.join(", ");

                            check_cfgs.push(format!("cfg({name}, values({values}))"))
                        }
                    }
                }

                check_cfgs.sort_unstable();
                if !sess.check_config.exhaustive_names && sess.check_config.exhaustive_values {
                    println_info!("cfg(any())");
                }
                for check_cfg in check_cfgs {
                    println_info!("{check_cfg}");
                }
            }
            CallingConventions => {
                let calling_conventions = rustc_abi::all_names();
                println_info!("{}", calling_conventions.join("\n"));
            }
            BackendHasMnemonic => {
                let has_mnemonic: bool =
                    codegen_backend.has_mnemonic(sess, req.arg.as_ref().unwrap());
                println_info!("{has_mnemonic}");
            }
            BackendHasZstd => {
                let has_zstd: bool = codegen_backend.has_zstd();
                println_info!("{has_zstd}");
            }
            RelocationModels
            | CodeModels
            | TlsModels
            | TargetCPUs
            | StackProtectorStrategies
            | TargetFeatures => {
                codegen_backend.print(req, &mut crate_info, sess);
            }
            // Any output here interferes with Cargo's parsing of other printed output
            NativeStaticLibs => {}
            LinkArgs => {}
            SplitDebuginfo => {
                use rustc_target::spec::SplitDebuginfo::{Off, Packed, Unpacked};

                for split in &[Off, Packed, Unpacked] {
                    if sess.target.options.supported_split_debuginfo.contains(split) {
                        println_info!("{split}");
                    }
                }
            }
            DeploymentTarget => {
                if sess.target.is_like_darwin {
                    println_info!(
                        "{}={}",
                        rustc_target::spec::apple::deployment_target_env_var(&sess.target.os),
                        sess.apple_deployment_target().fmt_pretty(),
                    )
                } else {
                    sess.dcx().fatal("only Apple targets currently support deployment version info")
                }
            }
            SupportedCrateTypes => {
                let supported_crate_types = CrateType::all()
                    .iter()
                    .filter(|(_, crate_type)| !invalid_output_for_target(sess, *crate_type))
                    .filter(|(_, crate_type)| *crate_type != CrateType::Sdylib)
                    .map(|(crate_type_sym, _)| *crate_type_sym)
                    .collect::<BTreeSet<_>>();
                for supported_crate_type in supported_crate_types {
                    println_info!("{}", supported_crate_type.as_str());
                }
            }
        }

        req.out.overwrite(&crate_info, sess);
    }
    Compilation::Stop
}
