use std::collections::{BTreeMap, BTreeSet};

use rustc_codegen_ssa::traits::{CodegenBackend, TargetCpuInfo};
use rustc_feature::find_gated_cfg;
// This avoids a false positive with `-Wunused_crate_dependencies`.
// `rust_index` isn't used in this crate's code, but it must be named in the
// `Cargo.toml` for the `rustc_randomized_layouts` feature.
use rustc_index as _;
use rustc_interface::passes;
use rustc_interface::passes::collect_crate_types;
use rustc_session::Session;
use rustc_session::config::{PrintKind, PrintOutputFormat, PrintRequest};
use rustc_session::output::invalid_output_for_target;
use rustc_span::DUMMY_SP;
use rustc_structures::CrateType;
use rustc_target::json::ToJson;
use rustc_target::spec::{RelocModel, Target, TargetTuple, TlsModel};
use serde_json::json;

use crate::{Compilation, parse_crate_attrs};

enum PrintValue {
    Bool(bool),
    Str(String),
    List(Vec<String>),
    Json(serde_json::Value),
}

impl PrintValue {
    fn format(self, req: &PrintRequest) -> String {
        match req.format {
            PrintOutputFormat::Default => match self {
                PrintValue::Bool(b) => format!("{b}\n"),
                PrintValue::Str(s) => format!("{s}\n"),
                PrintValue::List(items) => items.iter().map(|i| format!("{i}\n")).collect(),
                PrintValue::Json(v) => format!("{}\n", serde_json::to_string_pretty(&v).unwrap()),
            },
            PrintOutputFormat::Json => {
                let value = match self {
                    PrintValue::Bool(b) => json!(b),
                    PrintValue::Str(s) => json!(s),
                    PrintValue::List(items) => json!(items),
                    PrintValue::Json(v) => v,
                };
                let out = serde_json::to_string(&json!({
                    "message-type": "print",
                    "id": req.kind.name(),
                    "value": value,
                }))
                .expect("serializing PrintValue to JSON cannot fail");
                out + "\n"
            }
        }
    }
}

macro_rules! push_info {
    ($out:expr, $req:expr, @bool $v:expr) => {
        $out.push_str(&PrintValue::Bool($v).format($req))
    };
    ($out:expr, $req:expr, @vec $v:expr) => {
        $out.push_str(&PrintValue::List($v.iter().map(|x| x.to_string()).collect()).format($req))
    };
    ($out:expr, $req:expr, @json $v:expr) => {
        $out.push_str(&PrintValue::Json(serde_json::to_value(&$v).unwrap()).format($req))
    };
    ($out:expr, $req:expr, @string $($arg:tt)*) => {
        $out.push_str(&PrintValue::Str(format!($($arg)*)).format($req))
    };
}

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
        macro_rules! println_info {
            ($($t:tt)*) => { push_info!(crate_info, req, $($t)*) };
        }

        match req.kind {
            TargetList => {
                let mut targets = rustc_target::spec::TARGETS.to_vec();
                targets.sort_unstable();
                println_info!(@vec targets);
            }
            HostTuple => println_info!(@string "{}", rustc_session::config::host_tuple()),
            WasmProcMacroTuple => println_info!(@string "{}", sess.wasm_proc_macro_tuple),
            Sysroot => println_info!(@string "{}", sess.opts.sysroot.path().display()),
            TargetLibdir => println_info!(@string "{}", sess.target_tlib_path.dir.display()),
            TargetSpecJson => {
                println_info!(@json sess.target.to_json());
            }
            TargetSpecJsonSchema => {
                let schema = rustc_target::spec::json_schema();
                println_info!(@json schema);
            }
            AllTargetSpecsJson => {
                let mut targets = BTreeMap::new();
                for name in rustc_target::spec::TARGETS {
                    let triple = TargetTuple::from_tuple(name);
                    let target = Target::expect_builtin(&triple);
                    targets.insert(name, target.to_json());
                }
                println_info!(@json targets);
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
                let file_names: Vec<String> = crate_types
                    .iter()
                    .map(|&style| {
                        let fname = rustc_session::output::filename_for_input(
                            sess, style, crate_name, &t_outputs,
                        );
                        fname.as_path().file_name().unwrap().to_string_lossy().into_owned()
                    })
                    .collect();
                println_info!(@vec file_names);
            }
            CrateName => {
                let Some(attrs) = attrs.as_ref() else {
                    // no crate attributes, print out an error and exit
                    return Compilation::Continue;
                };
                println_info!(@string "{}", passes::get_crate_name(sess, attrs));
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
                let mut levels = Vec::new();
                for lint in lint_store.get_lints() {
                    if let Some(feature_symbol) = lint.feature_gate
                        && !features.enabled(feature_symbol)
                    {
                        // lint is unstable and feature gate isn't active, don't print
                        continue;
                    }
                    let level = builder.lint_level_spec(lint).level();
                    levels.push(format!("{}={}", lint.name_lower(), level.as_str()));
                }
                println_info!(@vec levels);
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
                println_info!(@vec cfgs);
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
                    check_cfgs.insert(0, "cfg(any())".to_string());
                }
                println_info!(@vec check_cfgs);
            }
            CallingConventions => {
                let calling_conventions = rustc_abi::all_names();
                println_info!(@vec calling_conventions);
            }
            BackendHasMnemonic => {
                let has_mnemonic: bool =
                    codegen_backend.has_mnemonic(sess, req.arg.as_ref().unwrap());
                println_info!(@bool has_mnemonic);
            }
            BackendHasZstd => {
                let has_zstd: bool = codegen_backend.has_zstd();
                println_info!(@bool has_zstd);
            }
            RelocationModels
            | CodeModels
            | TlsModels
            | TargetCPUs
            | StackProtectorStrategies
            | TargetFeatures => {
                print_backend_options(codegen_backend, sess, req, &mut crate_info);
            }
            // Any output here interferes with Cargo's parsing of other printed output
            NativeStaticLibs => {}
            LinkArgs => {}
            SplitDebuginfo => {
                use rustc_target::spec::SplitDebuginfo::{Off, Packed, Unpacked};

                let splits: Vec<_> = [Off, Packed, Unpacked]
                    .into_iter()
                    .filter(|split| sess.target.options.supported_split_debuginfo.contains(split))
                    .collect();
                println_info!(@vec splits);
            }
            DeploymentTarget => {
                if sess.target.is_like_darwin {
                    println_info!(
                        @string
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
                    .map(|(crate_type_sym, _)| crate_type_sym.as_str())
                    .collect::<BTreeSet<_>>();
                println_info!(@vec supported_crate_types);
            }
        }

        req.out.overwrite(&crate_info, sess);
    }
    Compilation::Stop
}

use std::fmt::Write;

fn print_model_list(req: &PrintRequest, out: &mut String, title: &str, names: &[&str]) {
    match req.format {
        PrintOutputFormat::Default => {
            writeln!(out, "Available {title}:").unwrap();
            for name in names {
                writeln!(out, "    {name}").unwrap();
            }
            writeln!(out).unwrap();
        }
        PrintOutputFormat::Json => push_info!(out, req, @vec names),
    }
}

const STACK_PROTECTOR_STRATEGIES: &[(&str, &str)] = &[
    ("all", "Generate stack canaries in all functions."),
    (
        "strong",
        r"Generate stack canaries in a function if it either:
- has a local variable of `[T; N]` type, regardless of `T` and `N`
- takes the address of a local variable.

  (Note that a local variable being borrowed is not equivalent to its
  address being taken: e.g. some borrows may be removed by optimization,
  while by-value argument passing may be implemented with reference to a
  local stack variable in the ABI.)",
    ),
    (
        "basic",
        r"Generate stack canaries in functions with local variables of `[T; N]`
type, where `T` is byte-sized and `N` >= 8.",
    ),
    ("none", "Do not generate stack canaries."),
];

fn print_backend_options(
    codegen_backend: &dyn CodegenBackend,
    sess: &Session,
    req: &PrintRequest,
    out: &mut String,
) {
    match req.kind {
        PrintKind::RelocationModels => {
            let names: Vec<&str> =
                RelocModel::ALL.iter().map(RelocModel::desc).chain(["default"]).collect();
            print_model_list(req, out, "relocation models", &names);
        }
        PrintKind::CodeModels => {
            print_model_list(
                req,
                out,
                "code models",
                &["tiny", "small", "kernel", "medium", "large"],
            );
        }
        PrintKind::TlsModels => {
            let names: Vec<&str> = TlsModel::ALL.iter().map(TlsModel::desc).collect();
            print_model_list(req, out, "TLS models", &names);
        }
        PrintKind::StackProtectorStrategies => match req.format {
            PrintOutputFormat::Default => {
                writeln!(out, "Available stack protector strategies:").unwrap();
                for (name, description) in STACK_PROTECTOR_STRATEGIES {
                    writeln!(out, "    {name}").unwrap();
                    for line in description.lines() {
                        if line.is_empty() {
                            writeln!(out).unwrap();
                        } else {
                            writeln!(out, "        {line}").unwrap();
                        }
                    }
                    writeln!(out).unwrap();
                }
            }
            PrintOutputFormat::Json => {
                let strategies: BTreeMap<_, _> =
                    STACK_PROTECTOR_STRATEGIES.iter().copied().collect();
                push_info!(out, req, @json strategies);
            }
        },
        PrintKind::TargetCPUs => {
            let cpus = codegen_backend.printable_target_cpus(sess);
            match req.format {
                PrintOutputFormat::Default => {
                    let max_name_width = cpus.iter().map(|cpu| cpu.name.len()).max().unwrap_or(0);
                    writeln!(out, "Available CPUs for this target:").unwrap();
                    for TargetCpuInfo { name, remark } in &cpus {
                        match remark {
                            // Only pad the CPU name if there is a remark to print after it.
                            Some(remark) => {
                                writeln!(out, "    {name:<max_name_width$} - {remark}").unwrap()
                            }
                            None => writeln!(out, "    {name}").unwrap(),
                        }
                    }
                }
                PrintOutputFormat::Json => {
                    let json_cpus: Vec<serde_json::Value> = cpus
                        .iter()
                        .map(|TargetCpuInfo { name, remark }| {
                            serde_json::json!({
                                "name": name,
                                "remark": remark,
                            })
                        })
                        .collect();
                    push_info!(out, req, @json json_cpus)
                }
            }
        }
        PrintKind::TargetFeatures => {
            let info = codegen_backend.printable_target_features(sess);
            match req.format {
                PrintOutputFormat::Default => {
                    let max_feature_len = info
                        .backend_features
                        .iter()
                        .chain(info.rustc_features.iter())
                        .map(|(feature, _desc)| feature.len())
                        .max()
                        .unwrap_or(0);

                    writeln!(out, "Features supported by rustc for this target:").unwrap();
                    for (feature, desc) in &info.rustc_features {
                        writeln!(out, "    {feature:max_feature_len$} - {desc}.").unwrap();
                    }
                    writeln!(out, "\nCode-generation features supported by LLVM for this target:")
                        .unwrap();
                    for (feature, desc) in &info.backend_features {
                        writeln!(out, "    {feature:max_feature_len$} - {desc}.").unwrap();
                    }
                    if info.backend_features.is_empty() {
                        writeln!(
                            out,
                            "    Target features listing is not supported by this LLVM version."
                        )
                        .unwrap();
                    }
                    writeln!(out, "\nUse +feature to enable a feature, or -feature to disable it.")
                        .unwrap();
                    writeln!(out, "For example, rustc -C target-cpu=mycpu -C target-feature=+feature1,-feature2\n").unwrap();
                    writeln!(
                        out,
                        "Code-generation features cannot be used in cfg or #[target_feature],"
                    )
                    .unwrap();
                    writeln!(
                        out,
                        "and may be renamed or removed in a future version of LLVM or rustc.\n"
                    )
                    .unwrap();
                }
                PrintOutputFormat::Json => {
                    let to_json = |features: &[(&str, &str)]| -> Vec<serde_json::Value> {
                        features
                            .iter()
                            .map(|(name, description)| json!({ "name": name, "description": description }))
                            .collect()
                    };
                    push_info!(out, req, @json json!({
                        "rustc_features": to_json(&info.rustc_features),
                        "backend_features": to_json(&info.backend_features),
                    }));
                }
            }
        }
        _ => unreachable!(),
    }
}
