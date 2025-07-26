use std::collections::BTreeMap;
use std::str::FromStr;

use rustc_abi::{Align, AlignFromBytesError};

use super::crt_objects::CrtObjects;
use super::{
    BinaryFormat, CodeModel, DebuginfoKind, FloatAbi, FramePointer, LinkArgsCli,
    LinkSelfContainedComponents, LinkSelfContainedDefault, LinkerFlavorCli, LldFlavor,
    MergeFunctions, PanicStrategy, RelocModel, RelroLevel, RustcAbi, SanitizerSet,
    SmallDataThresholdSupport, SplitDebuginfo, StackProbeType, StaticCow, SymbolVisibility, Target,
    TargetKind, TargetOptions, TargetWarnings, TlsModel,
};
use crate::json::{Json, ToJson};
use crate::spec::AbiMap;

impl Target {
    /// Loads a target descriptor from a JSON object.
    pub fn from_json(json: &str) -> Result<(Target, TargetWarnings), String> {
        let json_deserializer = &mut serde_json::Deserializer::from_str(json);

        let json: TargetSpecJson =
            serde_path_to_error::deserialize(json_deserializer).map_err(|err| err.to_string())?;

        let mut base = Target {
            llvm_target: json.llvm_target,
            metadata: Default::default(),
            pointer_width: json.target_pointer_width,
            data_layout: json.data_layout,
            arch: json.arch,
            options: Default::default(),
        };

        // FIXME: This doesn't properly validate anything and just ignores the data if it's invalid.
        // That's okay for now, the only use of this is when generating docs, which we don't do for
        // custom targets.
        if let Some(metadata) = json.metadata {
            base.metadata.description = metadata.description;
            base.metadata.tier = metadata.tier.filter(|tier| (1..=3).contains(tier));
            base.metadata.host_tools = metadata.host_tools;
            base.metadata.std = metadata.std;
        }

        let alignment_error = |field_name: &str, error: AlignFromBytesError| -> String {
            let msg = match error {
                AlignFromBytesError::NotPowerOfTwo(_) => "not a power of 2 number of bytes",
                AlignFromBytesError::TooLarge(_) => "too large",
            };
            format!("`{}` bits is not a valid value for {field_name}: {msg}", error.align() * 8)
        };

        macro_rules! forward {
            ($name:ident) => {
                if let Some($name) = json.$name {
                    base.$name = $name;
                }
            };
        }
        macro_rules! forward_opt {
            ($name:ident) => {
                if let Some($name) = json.$name {
                    base.$name = Some($name);
                }
            };
        }

        if let Some(target_endian) = json.target_endian {
            base.endian = target_endian.0;
        }

        forward!(frame_pointer);
        forward!(c_int_width);
        forward_opt!(c_enum_min_bits); // if None, matches c_int_width
        forward!(os);
        forward!(env);
        forward!(abi);
        forward!(vendor);
        forward_opt!(linker);
        forward!(linker_flavor_json);
        forward!(lld_flavor_json);
        forward!(linker_is_gnu_json);
        forward!(pre_link_objects);
        forward!(post_link_objects);
        forward!(pre_link_objects_self_contained);
        forward!(post_link_objects_self_contained);

        // Deserializes the backwards-compatible variants of `-Clink-self-contained`
        if let Some(link_self_contained) = json.link_self_contained_backwards_compatible {
            base.link_self_contained = link_self_contained;
        }
        // Deserializes the components variant of `-Clink-self-contained`
        if let Some(link_self_contained) = json.link_self_contained {
            let components = link_self_contained
                .components
                .into_iter()
                .fold(LinkSelfContainedComponents::empty(), |a, b| a | b);
            base.link_self_contained = LinkSelfContainedDefault::WithComponents(components);
        }

        forward!(pre_link_args_json);
        forward!(late_link_args_json);
        forward!(late_link_args_dynamic_json);
        forward!(late_link_args_static_json);
        forward!(post_link_args_json);
        forward_opt!(link_script);

        if let Some(link_env) = json.link_env {
            for s in link_env {
                if let [k, v] = *s.split('=').collect::<Vec<_>>() {
                    base.link_env.to_mut().push((k.to_string().into(), v.to_string().into()))
                } else {
                    return Err(format!("link-env value '{s}' must be of the pattern 'KEY=VALUE'"));
                }
            }
        }

        forward!(link_env_remove);
        forward!(asm_args);
        forward!(cpu);
        forward!(need_explicit_cpu);
        forward!(features);
        forward!(dynamic_linking);
        forward_opt!(direct_access_external_data);
        forward!(dll_tls_export);
        forward!(only_cdylib);
        forward!(executables);
        forward!(relocation_model);
        forward_opt!(code_model);
        forward!(tls_model);
        forward!(disable_redzone);
        forward!(function_sections);
        forward!(dll_prefix);
        forward!(dll_suffix);
        forward!(exe_suffix);
        forward!(staticlib_prefix);
        forward!(staticlib_suffix);

        if let Some(target_family) = json.target_family {
            match target_family {
                TargetFamiliesJson::Array(families) => base.families = families,
                TargetFamiliesJson::String(family) => base.families = vec![family].into(),
            }
        }

        forward!(abi_return_struct_as_int);
        forward!(is_like_aix);
        forward!(is_like_darwin);
        forward!(is_like_solaris);
        forward!(is_like_windows);
        forward!(is_like_msvc);
        forward!(is_like_wasm);
        forward!(is_like_android);
        forward!(is_like_vexos);
        forward!(binary_format);
        forward!(default_dwarf_version);
        forward!(allows_weak_linkage);
        forward!(has_rpath);
        forward!(no_default_libraries);
        forward!(position_independent_executables);
        forward!(static_position_independent_executables);
        forward!(plt_by_default);
        forward!(relro_level);
        forward!(archive_format);
        forward!(allow_asm);
        forward!(main_needs_argc_argv);
        forward!(has_thread_local);
        forward!(obj_is_bitcode);
        forward_opt!(max_atomic_width);
        forward_opt!(min_atomic_width);
        forward!(atomic_cas);
        forward!(panic_strategy);
        forward!(crt_static_allows_dylibs);
        forward!(crt_static_default);
        forward!(crt_static_respected);
        forward!(stack_probes);

        if let Some(min_global_align) = json.min_global_align {
            match Align::from_bits(min_global_align) {
                Ok(align) => base.min_global_align = Some(align),
                Err(e) => return Err(alignment_error("min-global-align", e)),
            }
        }

        forward_opt!(default_codegen_units);
        forward_opt!(default_codegen_backend);
        forward!(trap_unreachable);
        forward!(requires_lto);
        forward!(singlethread);
        forward!(no_builtins);
        forward_opt!(default_visibility);
        forward!(emit_debug_gdb_scripts);
        forward!(requires_uwtable);
        forward!(default_uwtable);
        forward!(simd_types_indirect);
        forward!(limit_rdylib_exports);
        forward_opt!(override_export_symbols);
        forward!(merge_functions);
        forward!(mcount);
        forward_opt!(llvm_mcount_intrinsic);
        forward!(llvm_abiname);
        forward_opt!(llvm_floatabi);
        forward_opt!(rustc_abi);
        forward!(relax_elf_relocations);
        forward!(llvm_args);
        forward!(use_ctors_section);
        forward!(eh_frame_header);
        forward!(has_thumb_interworking);
        forward!(debuginfo_kind);
        forward!(split_debuginfo);
        forward!(supported_split_debuginfo);

        if let Some(supported_sanitizers) = json.supported_sanitizers {
            base.supported_sanitizers =
                supported_sanitizers.into_iter().fold(SanitizerSet::empty(), |a, b| a | b);
        }

        forward!(generate_arange_section);
        forward!(supports_stack_protector);
        forward!(small_data_threshold_support);
        forward!(entry_name);
        forward!(supports_xray);

        // we're going to run `update_from_cli`, but that won't change the target's AbiMap
        // FIXME: better factor the Target definition so we enforce this on a type level
        let abi_map = AbiMap::from_target(&base);
        if let Some(entry_abi) = json.entry_abi {
            base.options.entry_abi = abi_map.canonize_abi(entry_abi.0, false).unwrap();
        }

        base.update_from_cli();
        base.check_consistency(TargetKind::Json)?;

        Ok((base, TargetWarnings { unused_fields: vec![] }))
    }
}

impl ToJson for Target {
    fn to_json(&self) -> Json {
        let mut d = serde_json::Map::new();
        let default: TargetOptions = Default::default();
        let mut target = self.clone();
        target.update_to_cli();

        macro_rules! target_val {
            ($attr:ident) => {
                target_val!($attr, (stringify!($attr)).replace("_", "-"))
            };
            ($attr:ident, $json_name:expr) => {{
                let name = $json_name;
                d.insert(name.into(), target.$attr.to_json());
            }};
        }

        macro_rules! target_option_val {
            ($attr:ident) => {{ target_option_val!($attr, (stringify!($attr)).replace("_", "-")) }};
            ($attr:ident, $json_name:expr) => {{
                let name = $json_name;
                if default.$attr != target.$attr {
                    d.insert(name.into(), target.$attr.to_json());
                }
            }};
            (link_args - $attr:ident, $json_name:expr) => {{
                let name = $json_name;
                if default.$attr != target.$attr {
                    let obj = target
                        .$attr
                        .iter()
                        .map(|(k, v)| (k.desc().to_string(), v.clone()))
                        .collect::<BTreeMap<_, _>>();
                    d.insert(name.to_string(), obj.to_json());
                }
            }};
            (env - $attr:ident) => {{
                let name = (stringify!($attr)).replace("_", "-");
                if default.$attr != target.$attr {
                    let obj = target
                        .$attr
                        .iter()
                        .map(|&(ref k, ref v)| format!("{k}={v}"))
                        .collect::<Vec<_>>();
                    d.insert(name, obj.to_json());
                }
            }};
        }

        target_val!(llvm_target);
        target_val!(metadata);
        target_val!(pointer_width, "target-pointer-width");
        target_val!(arch);
        target_val!(data_layout);

        target_option_val!(endian, "target-endian");
        target_option_val!(c_int_width, "target-c-int-width");
        target_option_val!(os);
        target_option_val!(env);
        target_option_val!(abi);
        target_option_val!(vendor);
        target_option_val!(linker);
        target_option_val!(linker_flavor_json, "linker-flavor");
        target_option_val!(lld_flavor_json, "lld-flavor");
        target_option_val!(linker_is_gnu_json, "linker-is-gnu");
        target_option_val!(pre_link_objects);
        target_option_val!(post_link_objects);
        target_option_val!(pre_link_objects_self_contained, "pre-link-objects-fallback");
        target_option_val!(post_link_objects_self_contained, "post-link-objects-fallback");
        target_option_val!(link_args - pre_link_args_json, "pre-link-args");
        target_option_val!(link_args - late_link_args_json, "late-link-args");
        target_option_val!(link_args - late_link_args_dynamic_json, "late-link-args-dynamic");
        target_option_val!(link_args - late_link_args_static_json, "late-link-args-static");
        target_option_val!(link_args - post_link_args_json, "post-link-args");
        target_option_val!(link_script);
        target_option_val!(env - link_env);
        target_option_val!(link_env_remove);
        target_option_val!(asm_args);
        target_option_val!(cpu);
        target_option_val!(need_explicit_cpu);
        target_option_val!(features);
        target_option_val!(dynamic_linking);
        target_option_val!(direct_access_external_data);
        target_option_val!(dll_tls_export);
        target_option_val!(only_cdylib);
        target_option_val!(executables);
        target_option_val!(relocation_model);
        target_option_val!(code_model);
        target_option_val!(tls_model);
        target_option_val!(disable_redzone);
        target_option_val!(frame_pointer);
        target_option_val!(function_sections);
        target_option_val!(dll_prefix);
        target_option_val!(dll_suffix);
        target_option_val!(exe_suffix);
        target_option_val!(staticlib_prefix);
        target_option_val!(staticlib_suffix);
        target_option_val!(families, "target-family");
        target_option_val!(abi_return_struct_as_int);
        target_option_val!(is_like_aix);
        target_option_val!(is_like_darwin);
        target_option_val!(is_like_solaris);
        target_option_val!(is_like_windows);
        target_option_val!(is_like_msvc);
        target_option_val!(is_like_wasm);
        target_option_val!(is_like_android);
        target_option_val!(is_like_vexos);
        target_option_val!(binary_format);
        target_option_val!(default_dwarf_version);
        target_option_val!(allows_weak_linkage);
        target_option_val!(has_rpath);
        target_option_val!(no_default_libraries);
        target_option_val!(position_independent_executables);
        target_option_val!(static_position_independent_executables);
        target_option_val!(plt_by_default);
        target_option_val!(relro_level);
        target_option_val!(archive_format);
        target_option_val!(allow_asm);
        target_option_val!(main_needs_argc_argv);
        target_option_val!(has_thread_local);
        target_option_val!(obj_is_bitcode);
        target_option_val!(min_atomic_width);
        target_option_val!(max_atomic_width);
        target_option_val!(atomic_cas);
        target_option_val!(panic_strategy);
        target_option_val!(crt_static_allows_dylibs);
        target_option_val!(crt_static_default);
        target_option_val!(crt_static_respected);
        target_option_val!(stack_probes);
        target_option_val!(min_global_align);
        target_option_val!(default_codegen_units);
        target_option_val!(default_codegen_backend);
        target_option_val!(trap_unreachable);
        target_option_val!(requires_lto);
        target_option_val!(singlethread);
        target_option_val!(no_builtins);
        target_option_val!(default_visibility);
        target_option_val!(emit_debug_gdb_scripts);
        target_option_val!(requires_uwtable);
        target_option_val!(default_uwtable);
        target_option_val!(simd_types_indirect);
        target_option_val!(limit_rdylib_exports);
        target_option_val!(override_export_symbols);
        target_option_val!(merge_functions);
        target_option_val!(mcount, "target-mcount");
        target_option_val!(llvm_mcount_intrinsic);
        target_option_val!(llvm_abiname);
        target_option_val!(llvm_floatabi);
        target_option_val!(rustc_abi);
        target_option_val!(relax_elf_relocations);
        target_option_val!(llvm_args);
        target_option_val!(use_ctors_section);
        target_option_val!(eh_frame_header);
        target_option_val!(has_thumb_interworking);
        target_option_val!(debuginfo_kind);
        target_option_val!(split_debuginfo);
        target_option_val!(supported_split_debuginfo);
        target_option_val!(supported_sanitizers);
        target_option_val!(c_enum_min_bits);
        target_option_val!(generate_arange_section);
        target_option_val!(supports_stack_protector);
        target_option_val!(small_data_threshold_support);
        target_option_val!(entry_name);
        target_option_val!(entry_abi);
        target_option_val!(supports_xray);

        // Serializing `-Clink-self-contained` needs a dynamic key to support the
        // backwards-compatible variants.
        d.insert(self.link_self_contained.json_key().into(), self.link_self_contained.to_json());

        Json::Object(d)
    }
}

#[derive(serde_derive::Deserialize, schemars::JsonSchema)]
struct LinkSelfContainedComponentsWrapper {
    components: Vec<LinkSelfContainedComponents>,
}

#[derive(serde_derive::Deserialize, schemars::JsonSchema)]
#[serde(untagged)]
enum TargetFamiliesJson {
    Array(StaticCow<[StaticCow<str>]>),
    String(StaticCow<str>),
}

/// `Endian` is in `rustc_abi`, which doesn't have access to the macro and serde.
struct EndianWrapper(rustc_abi::Endian);
impl FromStr for EndianWrapper {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        rustc_abi::Endian::from_str(s).map(Self)
    }
}
crate::json::serde_deserialize_from_str!(EndianWrapper);
impl schemars::JsonSchema for EndianWrapper {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "Endian".into()
    }
    fn json_schema(_: &mut schemars::SchemaGenerator) -> schemars::Schema {
        schemars::json_schema! ({
            "type": "string",
            "enum": ["big", "little"]
        })
        .into()
    }
}

/// `ExternAbi` is in `rustc_abi`, which doesn't have access to the macro and serde.
struct ExternAbiWrapper(rustc_abi::ExternAbi);
impl FromStr for ExternAbiWrapper {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        rustc_abi::ExternAbi::from_str(s)
            .map(Self)
            .map_err(|_| format!("{s} is not a valid extern ABI"))
    }
}
crate::json::serde_deserialize_from_str!(ExternAbiWrapper);
impl schemars::JsonSchema for ExternAbiWrapper {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "ExternAbi".into()
    }
    fn json_schema(_: &mut schemars::SchemaGenerator) -> schemars::Schema {
        let all =
            rustc_abi::ExternAbi::ALL_VARIANTS.iter().map(|abi| abi.as_str()).collect::<Vec<_>>();
        schemars::json_schema! ({
            "type": "string",
            "enum": all,
        })
        .into()
    }
}

#[derive(serde_derive::Deserialize, schemars::JsonSchema)]
struct TargetSpecJsonMetadata {
    description: Option<StaticCow<str>>,
    tier: Option<u64>,
    host_tools: Option<bool>,
    std: Option<bool>,
}

#[derive(serde_derive::Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "kebab-case")]
// Ensure that all unexpected fields get turned into errors.
// This helps users stay up to date when the schema changes instead of silently
// ignoring their old values.
#[serde(deny_unknown_fields)]
struct TargetSpecJson {
    llvm_target: StaticCow<str>,
    target_pointer_width: u16,
    data_layout: StaticCow<str>,
    arch: StaticCow<str>,

    metadata: Option<TargetSpecJsonMetadata>,

    // options:
    target_endian: Option<EndianWrapper>,
    frame_pointer: Option<FramePointer>,
    #[serde(rename = "target-c-int-width")]
    c_int_width: Option<u16>,
    c_enum_min_bits: Option<u64>,
    os: Option<StaticCow<str>>,
    env: Option<StaticCow<str>>,
    abi: Option<StaticCow<str>>,
    vendor: Option<StaticCow<str>>,
    linker: Option<StaticCow<str>>,
    #[serde(rename = "linker-flavor")]
    linker_flavor_json: Option<LinkerFlavorCli>,
    #[serde(rename = "lld-flavor")]
    lld_flavor_json: Option<LldFlavor>,
    #[serde(rename = "linker-is-gnu")]
    linker_is_gnu_json: Option<bool>,
    #[serde(rename = "pre-link-objects")]
    pre_link_objects: Option<CrtObjects>,
    #[serde(rename = "post-link-objects")]
    post_link_objects: Option<CrtObjects>,
    #[serde(rename = "pre-link-objects-fallback")]
    pre_link_objects_self_contained: Option<CrtObjects>,
    #[serde(rename = "post-link-objects-fallback")]
    post_link_objects_self_contained: Option<CrtObjects>,
    #[serde(rename = "crt-objects-fallback")]
    link_self_contained_backwards_compatible: Option<LinkSelfContainedDefault>,
    link_self_contained: Option<LinkSelfContainedComponentsWrapper>,
    #[serde(rename = "pre-link-args")]
    pre_link_args_json: Option<LinkArgsCli>,
    #[serde(rename = "late-link-args")]
    late_link_args_json: Option<LinkArgsCli>,
    #[serde(rename = "late-link-args-dynamic")]
    late_link_args_dynamic_json: Option<LinkArgsCli>,
    #[serde(rename = "late-link-args-static")]
    late_link_args_static_json: Option<LinkArgsCli>,
    #[serde(rename = "post-link-args")]
    post_link_args_json: Option<LinkArgsCli>,
    link_script: Option<StaticCow<str>>,
    link_env: Option<Vec<StaticCow<str>>>,
    link_env_remove: Option<StaticCow<[StaticCow<str>]>>,
    asm_args: Option<StaticCow<[StaticCow<str>]>>,
    cpu: Option<StaticCow<str>>,
    need_explicit_cpu: Option<bool>,
    features: Option<StaticCow<str>>,
    dynamic_linking: Option<bool>,
    direct_access_external_data: Option<bool>,
    dll_tls_export: Option<bool>,
    only_cdylib: Option<bool>,
    executables: Option<bool>,
    relocation_model: Option<RelocModel>,
    code_model: Option<CodeModel>,
    tls_model: Option<TlsModel>,
    disable_redzone: Option<bool>,
    function_sections: Option<bool>,
    dll_prefix: Option<StaticCow<str>>,
    dll_suffix: Option<StaticCow<str>>,
    exe_suffix: Option<StaticCow<str>>,
    staticlib_prefix: Option<StaticCow<str>>,
    staticlib_suffix: Option<StaticCow<str>>,
    target_family: Option<TargetFamiliesJson>,
    abi_return_struct_as_int: Option<bool>,
    is_like_aix: Option<bool>,
    is_like_darwin: Option<bool>,
    is_like_solaris: Option<bool>,
    is_like_windows: Option<bool>,
    is_like_msvc: Option<bool>,
    is_like_wasm: Option<bool>,
    is_like_android: Option<bool>,
    is_like_vexos: Option<bool>,
    binary_format: Option<BinaryFormat>,
    default_dwarf_version: Option<u32>,
    allows_weak_linkage: Option<bool>,
    has_rpath: Option<bool>,
    no_default_libraries: Option<bool>,
    position_independent_executables: Option<bool>,
    static_position_independent_executables: Option<bool>,
    plt_by_default: Option<bool>,
    relro_level: Option<RelroLevel>,
    archive_format: Option<StaticCow<str>>,
    allow_asm: Option<bool>,
    main_needs_argc_argv: Option<bool>,
    has_thread_local: Option<bool>,
    obj_is_bitcode: Option<bool>,
    max_atomic_width: Option<u64>,
    min_atomic_width: Option<u64>,
    atomic_cas: Option<bool>,
    panic_strategy: Option<PanicStrategy>,
    crt_static_allows_dylibs: Option<bool>,
    crt_static_default: Option<bool>,
    crt_static_respected: Option<bool>,
    stack_probes: Option<StackProbeType>,
    min_global_align: Option<u64>,
    default_codegen_units: Option<u64>,
    default_codegen_backend: Option<StaticCow<str>>,
    trap_unreachable: Option<bool>,
    requires_lto: Option<bool>,
    singlethread: Option<bool>,
    no_builtins: Option<bool>,
    default_visibility: Option<SymbolVisibility>,
    emit_debug_gdb_scripts: Option<bool>,
    requires_uwtable: Option<bool>,
    default_uwtable: Option<bool>,
    simd_types_indirect: Option<bool>,
    limit_rdylib_exports: Option<bool>,
    override_export_symbols: Option<StaticCow<[StaticCow<str>]>>,
    merge_functions: Option<MergeFunctions>,
    #[serde(rename = "target-mcount")]
    mcount: Option<StaticCow<str>>,
    llvm_mcount_intrinsic: Option<StaticCow<str>>,
    llvm_abiname: Option<StaticCow<str>>,
    llvm_floatabi: Option<FloatAbi>,
    rustc_abi: Option<RustcAbi>,
    relax_elf_relocations: Option<bool>,
    llvm_args: Option<StaticCow<[StaticCow<str>]>>,
    use_ctors_section: Option<bool>,
    eh_frame_header: Option<bool>,
    has_thumb_interworking: Option<bool>,
    debuginfo_kind: Option<DebuginfoKind>,
    split_debuginfo: Option<SplitDebuginfo>,
    supported_split_debuginfo: Option<StaticCow<[SplitDebuginfo]>>,
    supported_sanitizers: Option<Vec<SanitizerSet>>,
    generate_arange_section: Option<bool>,
    supports_stack_protector: Option<bool>,
    small_data_threshold_support: Option<SmallDataThresholdSupport>,
    entry_name: Option<StaticCow<str>>,
    supports_xray: Option<bool>,
    entry_abi: Option<ExternAbiWrapper>,
}

pub fn json_schema() -> schemars::Schema {
    schemars::schema_for!(TargetSpecJson)
}
