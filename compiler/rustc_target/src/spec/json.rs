use std::borrow::Cow;
use std::collections::BTreeMap;
use std::str::FromStr;

use rustc_abi::ExternAbi;
use serde_json::Value;

use super::{Target, TargetKind, TargetOptions, TargetWarnings};
use crate::json::{Json, ToJson};
use crate::spec::AbiMap;

impl Target {
    /// Loads a target descriptor from a JSON object.
    pub fn from_json(obj: Json) -> Result<(Target, TargetWarnings), String> {
        // While ugly, this code must remain this way to retain
        // compatibility with existing JSON fields and the internal
        // expected naming of the Target and TargetOptions structs.
        // To ensure compatibility is retained, the built-in targets
        // are round-tripped through this code to catch cases where
        // the JSON parser is not updated to match the structs.

        let mut obj = match obj {
            Value::Object(obj) => obj,
            _ => return Err("Expected JSON object for target")?,
        };

        let mut get_req_field = |name: &str| {
            obj.remove(name)
                .and_then(|j| j.as_str().map(str::to_string))
                .ok_or_else(|| format!("Field {name} in target specification is required"))
        };

        let mut base = Target {
            llvm_target: get_req_field("llvm-target")?.into(),
            metadata: Default::default(),
            pointer_width: get_req_field("target-pointer-width")?
                .parse::<u32>()
                .map_err(|_| "target-pointer-width must be an integer".to_string())?,
            data_layout: get_req_field("data-layout")?.into(),
            arch: get_req_field("arch")?.into(),
            options: Default::default(),
        };

        // FIXME: This doesn't properly validate anything and just ignores the data if it's invalid.
        // That's okay for now, the only use of this is when generating docs, which we don't do for
        // custom targets.
        if let Some(Json::Object(mut metadata)) = obj.remove("metadata") {
            base.metadata.description = metadata
                .remove("description")
                .and_then(|desc| desc.as_str().map(|desc| desc.to_owned().into()));
            base.metadata.tier = metadata
                .remove("tier")
                .and_then(|tier| tier.as_u64())
                .filter(|tier| (1..=3).contains(tier));
            base.metadata.host_tools =
                metadata.remove("host_tools").and_then(|host| host.as_bool());
            base.metadata.std = metadata.remove("std").and_then(|host| host.as_bool());
        }

        let mut incorrect_type = vec![];

        macro_rules! key {
            ($key_name:ident) => ( {
                let name = (stringify!($key_name)).replace("_", "-");
                if let Some(s) = obj.remove(&name).and_then(|s| s.as_str().map(str::to_string).map(Cow::from)) {
                    base.$key_name = s;
                }
            } );
            ($key_name:ident = $json_name:expr) => ( {
                let name = $json_name;
                if let Some(s) = obj.remove(name).and_then(|s| s.as_str().map(str::to_string).map(Cow::from)) {
                    base.$key_name = s;
                }
            } );
            ($key_name:ident, bool) => ( {
                let name = (stringify!($key_name)).replace("_", "-");
                if let Some(s) = obj.remove(&name).and_then(|b| b.as_bool()) {
                    base.$key_name = s;
                }
            } );
            ($key_name:ident = $json_name:expr, bool) => ( {
                let name = $json_name;
                if let Some(s) = obj.remove(name).and_then(|b| b.as_bool()) {
                    base.$key_name = s;
                }
            } );
            ($key_name:ident, u32) => ( {
                let name = (stringify!($key_name)).replace("_", "-");
                if let Some(s) = obj.remove(&name).and_then(|b| b.as_u64()) {
                    if s < 1 || s > 5 {
                        return Err("Not a valid DWARF version number".into());
                    }
                    base.$key_name = s as u32;
                }
            } );
            ($key_name:ident, Option<bool>) => ( {
                let name = (stringify!($key_name)).replace("_", "-");
                if let Some(s) = obj.remove(&name).and_then(|b| b.as_bool()) {
                    base.$key_name = Some(s);
                }
            } );
            ($key_name:ident, Option<u64>) => ( {
                let name = (stringify!($key_name)).replace("_", "-");
                if let Some(s) = obj.remove(&name).and_then(|b| b.as_u64()) {
                    base.$key_name = Some(s);
                }
            } );
            ($key_name:ident, Option<StaticCow<str>>) => ( {
                let name = (stringify!($key_name)).replace("_", "-");
                if let Some(s) = obj.remove(&name).and_then(|b| Some(b.as_str()?.to_string())) {
                    base.$key_name = Some(s.into());
                }
            } );
            ($key_name:ident, BinaryFormat) => ( {
                let name = (stringify!($key_name)).replace("_", "-");
                obj.remove(&name).and_then(|f| f.as_str().and_then(|s| {
                    match s.parse::<super::BinaryFormat>() {
                        Ok(binary_format) => base.$key_name = binary_format,
                        _ => return Some(Err(format!(
                            "'{s}' is not a valid value for binary_format. \
                            Use 'coff', 'elf', 'mach-o', 'wasm' or 'xcoff' "
                        ))),
                    }
                    Some(Ok(()))
                })).unwrap_or(Ok(()))
            } );
            ($key_name:ident, MergeFunctions) => ( {
                let name = (stringify!($key_name)).replace("_", "-");
                obj.remove(&name).and_then(|o| o.as_str().and_then(|s| {
                    match s.parse::<super::MergeFunctions>() {
                        Ok(mergefunc) => base.$key_name = mergefunc,
                        _ => return Some(Err(format!("'{}' is not a valid value for \
                                                      merge-functions. Use 'disabled', \
                                                      'trampolines', or 'aliases'.",
                                                      s))),
                    }
                    Some(Ok(()))
                })).unwrap_or(Ok(()))
            } );
            ($key_name:ident, FloatAbi) => ( {
                let name = (stringify!($key_name)).replace("_", "-");
                obj.remove(&name).and_then(|o| o.as_str().and_then(|s| {
                    match s.parse::<super::FloatAbi>() {
                        Ok(float_abi) => base.$key_name = Some(float_abi),
                        _ => return Some(Err(format!("'{}' is not a valid value for \
                                                      llvm-floatabi. Use 'soft' or 'hard'.",
                                                      s))),
                    }
                    Some(Ok(()))
                })).unwrap_or(Ok(()))
            } );
            ($key_name:ident, RustcAbi) => ( {
                let name = (stringify!($key_name)).replace("_", "-");
                obj.remove(&name).and_then(|o| o.as_str().and_then(|s| {
                    match s.parse::<super::RustcAbi>() {
                        Ok(rustc_abi) => base.$key_name = Some(rustc_abi),
                        _ => return Some(Err(format!(
                            "'{s}' is not a valid value for rustc-abi. \
                            Use 'x86-softfloat' or leave the field unset."
                        ))),
                    }
                    Some(Ok(()))
                })).unwrap_or(Ok(()))
            } );
            ($key_name:ident, RelocModel) => ( {
                let name = (stringify!($key_name)).replace("_", "-");
                obj.remove(&name).and_then(|o| o.as_str().and_then(|s| {
                    match s.parse::<super::RelocModel>() {
                        Ok(relocation_model) => base.$key_name = relocation_model,
                        _ => return Some(Err(format!("'{}' is not a valid relocation model. \
                                                      Run `rustc --print relocation-models` to \
                                                      see the list of supported values.", s))),
                    }
                    Some(Ok(()))
                })).unwrap_or(Ok(()))
            } );
            ($key_name:ident, CodeModel) => ( {
                let name = (stringify!($key_name)).replace("_", "-");
                obj.remove(&name).and_then(|o| o.as_str().and_then(|s| {
                    match s.parse::<super::CodeModel>() {
                        Ok(code_model) => base.$key_name = Some(code_model),
                        _ => return Some(Err(format!("'{}' is not a valid code model. \
                                                      Run `rustc --print code-models` to \
                                                      see the list of supported values.", s))),
                    }
                    Some(Ok(()))
                })).unwrap_or(Ok(()))
            } );
            ($key_name:ident, TlsModel) => ( {
                let name = (stringify!($key_name)).replace("_", "-");
                obj.remove(&name).and_then(|o| o.as_str().and_then(|s| {
                    match s.parse::<super::TlsModel>() {
                        Ok(tls_model) => base.$key_name = tls_model,
                        _ => return Some(Err(format!("'{}' is not a valid TLS model. \
                                                      Run `rustc --print tls-models` to \
                                                      see the list of supported values.", s))),
                    }
                    Some(Ok(()))
                })).unwrap_or(Ok(()))
            } );
            ($key_name:ident, SmallDataThresholdSupport) => ( {
                obj.remove("small-data-threshold-support").and_then(|o| o.as_str().and_then(|s| {
                    match s.parse::<super::SmallDataThresholdSupport>() {
                        Ok(support) => base.small_data_threshold_support = support,
                        _ => return Some(Err(format!("'{s}' is not a valid value for small-data-threshold-support."))),
                    }
                    Some(Ok(()))
                })).unwrap_or(Ok(()))
            } );
            ($key_name:ident, PanicStrategy) => ( {
                let name = (stringify!($key_name)).replace("_", "-");
                obj.remove(&name).and_then(|o| o.as_str().and_then(|s| {
                    match s {
                        "unwind" => base.$key_name = super::PanicStrategy::Unwind,
                        "abort" => base.$key_name = super::PanicStrategy::Abort,
                        _ => return Some(Err(format!("'{}' is not a valid value for \
                                                      panic-strategy. Use 'unwind' or 'abort'.",
                                                     s))),
                }
                Some(Ok(()))
            })).unwrap_or(Ok(()))
            } );
            ($key_name:ident, RelroLevel) => ( {
                let name = (stringify!($key_name)).replace("_", "-");
                obj.remove(&name).and_then(|o| o.as_str().and_then(|s| {
                    match s.parse::<super::RelroLevel>() {
                        Ok(level) => base.$key_name = level,
                        _ => return Some(Err(format!("'{}' is not a valid value for \
                                                      relro-level. Use 'full', 'partial, or 'off'.",
                                                      s))),
                    }
                    Some(Ok(()))
                })).unwrap_or(Ok(()))
            } );
            ($key_name:ident, Option<SymbolVisibility>) => ( {
                let name = (stringify!($key_name)).replace("_", "-");
                obj.remove(&name).and_then(|o| o.as_str().and_then(|s| {
                    match s.parse::<super::SymbolVisibility>() {
                        Ok(level) => base.$key_name = Some(level),
                        _ => return Some(Err(format!("'{}' is not a valid value for \
                                                      symbol-visibility. Use 'hidden', 'protected, or 'interposable'.",
                                                      s))),
                    }
                    Some(Ok(()))
                })).unwrap_or(Ok(()))
            } );
            ($key_name:ident, DebuginfoKind) => ( {
                let name = (stringify!($key_name)).replace("_", "-");
                obj.remove(&name).and_then(|o| o.as_str().and_then(|s| {
                    match s.parse::<super::DebuginfoKind>() {
                        Ok(level) => base.$key_name = level,
                        _ => return Some(Err(
                            format!("'{s}' is not a valid value for debuginfo-kind. Use 'dwarf', \
                                  'dwarf-dsym' or 'pdb'.")
                        )),
                    }
                    Some(Ok(()))
                })).unwrap_or(Ok(()))
            } );
            ($key_name:ident, SplitDebuginfo) => ( {
                let name = (stringify!($key_name)).replace("_", "-");
                obj.remove(&name).and_then(|o| o.as_str().and_then(|s| {
                    match s.parse::<super::SplitDebuginfo>() {
                        Ok(level) => base.$key_name = level,
                        _ => return Some(Err(format!("'{}' is not a valid value for \
                                                      split-debuginfo. Use 'off' or 'dsymutil'.",
                                                      s))),
                    }
                    Some(Ok(()))
                })).unwrap_or(Ok(()))
            } );
            ($key_name:ident, list) => ( {
                let name = (stringify!($key_name)).replace("_", "-");
                if let Some(j) = obj.remove(&name) {
                    if let Some(v) = j.as_array() {
                        base.$key_name = v.iter()
                            .map(|a| a.as_str().unwrap().to_string().into())
                            .collect();
                    } else {
                        incorrect_type.push(name)
                    }
                }
            } );
            ($key_name:ident, opt_list) => ( {
                let name = (stringify!($key_name)).replace("_", "-");
                if let Some(j) = obj.remove(&name) {
                    if let Some(v) = j.as_array() {
                        base.$key_name = Some(v.iter()
                            .map(|a| a.as_str().unwrap().to_string().into())
                            .collect());
                    } else {
                        incorrect_type.push(name)
                    }
                }
            } );
            ($key_name:ident, fallible_list) => ( {
                let name = (stringify!($key_name)).replace("_", "-");
                obj.remove(&name).and_then(|j| {
                    if let Some(v) = j.as_array() {
                        match v.iter().map(|a| FromStr::from_str(a.as_str().unwrap())).collect() {
                            Ok(l) => { base.$key_name = l },
                            // FIXME: `fallible_list` can't re-use the `key!` macro for list
                            // elements and the error messages from that macro, so it has a bad
                            // generic message instead
                            Err(_) => return Some(Err(
                                format!("`{:?}` is not a valid value for `{}`", j, name)
                            )),
                        }
                    } else {
                        incorrect_type.push(name)
                    }
                    Some(Ok(()))
                }).unwrap_or(Ok(()))
            } );
            ($key_name:ident, optional) => ( {
                let name = (stringify!($key_name)).replace("_", "-");
                if let Some(o) = obj.remove(&name) {
                    base.$key_name = o
                        .as_str()
                        .map(|s| s.to_string().into());
                }
            } );
            ($key_name:ident = $json_name:expr, LldFlavor) => ( {
                let name = $json_name;
                obj.remove(name).and_then(|o| o.as_str().and_then(|s| {
                    if let Some(flavor) = super::LldFlavor::from_str(&s) {
                        base.$key_name = flavor;
                    } else {
                        return Some(Err(format!(
                            "'{}' is not a valid value for lld-flavor. \
                             Use 'darwin', 'gnu', 'link' or 'wasm'.",
                            s)))
                    }
                    Some(Ok(()))
                })).unwrap_or(Ok(()))
            } );
            ($key_name:ident = $json_name:expr, LinkerFlavorCli) => ( {
                let name = $json_name;
                obj.remove(name).and_then(|o| o.as_str().and_then(|s| {
                    match super::LinkerFlavorCli::from_str(s) {
                        Some(linker_flavor) => base.$key_name = linker_flavor,
                        _ => return Some(Err(format!("'{}' is not a valid value for linker-flavor. \
                                                      Use {}", s, super::LinkerFlavorCli::one_of()))),
                    }
                    Some(Ok(()))
                })).unwrap_or(Ok(()))
            } );
            ($key_name:ident, StackProbeType) => ( {
                let name = (stringify!($key_name)).replace("_", "-");
                obj.remove(&name).and_then(|o| match super::StackProbeType::from_json(&o) {
                    Ok(v) => {
                        base.$key_name = v;
                        Some(Ok(()))
                    },
                    Err(s) => Some(Err(
                        format!("`{:?}` is not a valid value for `{}`: {}", o, name, s)
                    )),
                }).unwrap_or(Ok(()))
            } );
            ($key_name:ident, SanitizerSet) => ( {
                let name = (stringify!($key_name)).replace("_", "-");
                if let Some(o) = obj.remove(&name) {
                    if let Some(a) = o.as_array() {
                        for s in a {
                            use super::SanitizerSet;
                            base.$key_name |= match s.as_str() {
                                Some("address") => SanitizerSet::ADDRESS,
                                Some("cfi") => SanitizerSet::CFI,
                                Some("dataflow") => SanitizerSet::DATAFLOW,
                                Some("kcfi") => SanitizerSet::KCFI,
                                Some("kernel-address") => SanitizerSet::KERNELADDRESS,
                                Some("leak") => SanitizerSet::LEAK,
                                Some("memory") => SanitizerSet::MEMORY,
                                Some("memtag") => SanitizerSet::MEMTAG,
                                Some("safestack") => SanitizerSet::SAFESTACK,
                                Some("shadow-call-stack") => SanitizerSet::SHADOWCALLSTACK,
                                Some("thread") => SanitizerSet::THREAD,
                                Some("hwaddress") => SanitizerSet::HWADDRESS,
                                Some(s) => return Err(format!("unknown sanitizer {}", s)),
                                _ => return Err(format!("not a string: {:?}", s)),
                            };
                        }
                    } else {
                        incorrect_type.push(name)
                    }
                }
                Ok::<(), String>(())
            } );
            ($key_name:ident, link_self_contained_components) => ( {
                // Skeleton of what needs to be parsed:
                //
                // ```
                // $name: {
                //     "components": [
                //         <array of strings>
                //     ]
                // }
                // ```
                let name = (stringify!($key_name)).replace("_", "-");
                if let Some(o) = obj.remove(&name) {
                    if let Some(o) = o.as_object() {
                        let component_array = o.get("components")
                            .ok_or_else(|| format!("{name}: expected a \
                                JSON object with a `components` field."))?;
                        let component_array = component_array.as_array()
                            .ok_or_else(|| format!("{name}.components: expected a JSON array"))?;
                        let mut components = super::LinkSelfContainedComponents::empty();
                        for s in component_array {
                            components |= match s.as_str() {
                                Some(s) => {
                                    super::LinkSelfContainedComponents::from_str(s)
                                        .ok_or_else(|| format!("unknown \
                                        `-Clink-self-contained` component: {s}"))?
                                },
                                _ => return Err(format!("not a string: {:?}", s)),
                            };
                        }
                        base.$key_name = super::LinkSelfContainedDefault::WithComponents(components);
                    } else {
                        incorrect_type.push(name)
                    }
                }
                Ok::<(), String>(())
            } );
            ($key_name:ident = $json_name:expr, link_self_contained_backwards_compatible) => ( {
                let name = $json_name;
                obj.remove(name).and_then(|o| o.as_str().and_then(|s| {
                    match s.parse::<super::LinkSelfContainedDefault>() {
                        Ok(lsc_default) => base.$key_name = lsc_default,
                        _ => return Some(Err(format!("'{}' is not a valid `-Clink-self-contained` default. \
                                                      Use 'false', 'true', 'musl' or 'mingw'", s))),
                    }
                    Some(Ok(()))
                })).unwrap_or(Ok(()))
            } );
            ($key_name:ident = $json_name:expr, link_objects) => ( {
                let name = $json_name;
                if let Some(val) = obj.remove(name) {
                    let obj = val.as_object().ok_or_else(|| format!("{}: expected a \
                        JSON object with fields per CRT object kind.", name))?;
                    let mut args = super::CrtObjects::new();
                    for (k, v) in obj {
                        let kind = super::LinkOutputKind::from_str(&k).ok_or_else(|| {
                            format!("{}: '{}' is not a valid value for CRT object kind. \
                                     Use '(dynamic,static)-(nopic,pic)-exe' or \
                                     '(dynamic,static)-dylib' or 'wasi-reactor-exe'", name, k)
                        })?;

                        let v = v.as_array().ok_or_else(||
                            format!("{}.{}: expected a JSON array", name, k)
                        )?.iter().enumerate()
                            .map(|(i,s)| {
                                let s = s.as_str().ok_or_else(||
                                    format!("{}.{}[{}]: expected a JSON string", name, k, i))?;
                                Ok(s.to_string().into())
                            })
                            .collect::<Result<Vec<_>, String>>()?;

                        args.insert(kind, v);
                    }
                    base.$key_name = args;
                }
            } );
            ($key_name:ident = $json_name:expr, link_args) => ( {
                let name = $json_name;
                if let Some(val) = obj.remove(name) {
                    let obj = val.as_object().ok_or_else(|| format!("{}: expected a \
                        JSON object with fields per linker-flavor.", name))?;
                    let mut args = super::LinkArgsCli::new();
                    for (k, v) in obj {
                        let flavor = super::LinkerFlavorCli::from_str(&k).ok_or_else(|| {
                            format!("{}: '{}' is not a valid value for linker-flavor. \
                                     Use 'em', 'gcc', 'ld' or 'msvc'", name, k)
                        })?;

                        let v = v.as_array().ok_or_else(||
                            format!("{}.{}: expected a JSON array", name, k)
                        )?.iter().enumerate()
                            .map(|(i,s)| {
                                let s = s.as_str().ok_or_else(||
                                    format!("{}.{}[{}]: expected a JSON string", name, k, i))?;
                                Ok(s.to_string().into())
                            })
                            .collect::<Result<Vec<_>, String>>()?;

                        args.insert(flavor, v);
                    }
                    base.$key_name = args;
                }
            } );
            ($key_name:ident, env) => ( {
                let name = (stringify!($key_name)).replace("_", "-");
                if let Some(o) = obj.remove(&name) {
                    if let Some(a) = o.as_array() {
                        for o in a {
                            if let Some(s) = o.as_str() {
                                if let [k, v] = *s.split('=').collect::<Vec<_>>() {
                                    base.$key_name
                                        .to_mut()
                                        .push((k.to_string().into(), v.to_string().into()))
                                }
                            }
                        }
                    } else {
                        incorrect_type.push(name)
                    }
                }
            } );
            ($key_name:ident, target_families) => ( {
                if let Some(value) = obj.remove("target-family") {
                    if let Some(v) = value.as_array() {
                        base.$key_name = v.iter()
                            .map(|a| a.as_str().unwrap().to_string().into())
                            .collect();
                    } else if let Some(v) = value.as_str() {
                        base.$key_name = vec![v.to_string().into()].into();
                    }
                }
            } );
        }

        if let Some(j) = obj.remove("target-endian") {
            if let Some(s) = j.as_str() {
                base.endian = s.parse()?;
            } else {
                incorrect_type.push("target-endian".into())
            }
        }

        if let Some(fp) = obj.remove("frame-pointer") {
            if let Some(s) = fp.as_str() {
                base.frame_pointer = s
                    .parse()
                    .map_err(|()| format!("'{s}' is not a valid value for frame-pointer"))?;
            } else {
                incorrect_type.push("frame-pointer".into())
            }
        }

        key!(c_int_width = "target-c-int-width");
        key!(c_enum_min_bits, Option<u64>); // if None, matches c_int_width
        key!(os);
        key!(env);
        key!(abi);
        key!(vendor);
        key!(linker, optional);
        key!(linker_flavor_json = "linker-flavor", LinkerFlavorCli)?;
        key!(lld_flavor_json = "lld-flavor", LldFlavor)?;
        key!(linker_is_gnu_json = "linker-is-gnu", bool);
        key!(pre_link_objects = "pre-link-objects", link_objects);
        key!(post_link_objects = "post-link-objects", link_objects);
        key!(pre_link_objects_self_contained = "pre-link-objects-fallback", link_objects);
        key!(post_link_objects_self_contained = "post-link-objects-fallback", link_objects);
        // Deserializes the backwards-compatible variants of `-Clink-self-contained`
        key!(
            link_self_contained = "crt-objects-fallback",
            link_self_contained_backwards_compatible
        )?;
        // Deserializes the components variant of `-Clink-self-contained`
        key!(link_self_contained, link_self_contained_components)?;
        key!(pre_link_args_json = "pre-link-args", link_args);
        key!(late_link_args_json = "late-link-args", link_args);
        key!(late_link_args_dynamic_json = "late-link-args-dynamic", link_args);
        key!(late_link_args_static_json = "late-link-args-static", link_args);
        key!(post_link_args_json = "post-link-args", link_args);
        key!(link_script, optional);
        key!(link_env, env);
        key!(link_env_remove, list);
        key!(asm_args, list);
        key!(cpu);
        key!(need_explicit_cpu, bool);
        key!(features);
        key!(dynamic_linking, bool);
        key!(direct_access_external_data, Option<bool>);
        key!(dll_tls_export, bool);
        key!(only_cdylib, bool);
        key!(executables, bool);
        key!(relocation_model, RelocModel)?;
        key!(code_model, CodeModel)?;
        key!(tls_model, TlsModel)?;
        key!(disable_redzone, bool);
        key!(function_sections, bool);
        key!(dll_prefix);
        key!(dll_suffix);
        key!(exe_suffix);
        key!(staticlib_prefix);
        key!(staticlib_suffix);
        key!(families, target_families);
        key!(abi_return_struct_as_int, bool);
        key!(is_like_aix, bool);
        key!(is_like_darwin, bool);
        key!(is_like_solaris, bool);
        key!(is_like_windows, bool);
        key!(is_like_msvc, bool);
        key!(is_like_wasm, bool);
        key!(is_like_android, bool);
        key!(binary_format, BinaryFormat)?;
        key!(default_dwarf_version, u32);
        key!(allows_weak_linkage, bool);
        key!(has_rpath, bool);
        key!(no_default_libraries, bool);
        key!(position_independent_executables, bool);
        key!(static_position_independent_executables, bool);
        key!(plt_by_default, bool);
        key!(relro_level, RelroLevel)?;
        key!(archive_format);
        key!(allow_asm, bool);
        key!(main_needs_argc_argv, bool);
        key!(has_thread_local, bool);
        key!(obj_is_bitcode, bool);
        key!(bitcode_llvm_cmdline);
        key!(max_atomic_width, Option<u64>);
        key!(min_atomic_width, Option<u64>);
        key!(atomic_cas, bool);
        key!(panic_strategy, PanicStrategy)?;
        key!(crt_static_allows_dylibs, bool);
        key!(crt_static_default, bool);
        key!(crt_static_respected, bool);
        key!(stack_probes, StackProbeType)?;
        key!(min_global_align, Option<u64>);
        key!(default_codegen_units, Option<u64>);
        key!(default_codegen_backend, Option<StaticCow<str>>);
        key!(trap_unreachable, bool);
        key!(requires_lto, bool);
        key!(singlethread, bool);
        key!(no_builtins, bool);
        key!(default_visibility, Option<SymbolVisibility>)?;
        key!(emit_debug_gdb_scripts, bool);
        key!(requires_uwtable, bool);
        key!(default_uwtable, bool);
        key!(simd_types_indirect, bool);
        key!(limit_rdylib_exports, bool);
        key!(override_export_symbols, opt_list);
        key!(merge_functions, MergeFunctions)?;
        key!(mcount = "target-mcount");
        key!(llvm_mcount_intrinsic, optional);
        key!(llvm_abiname);
        key!(llvm_floatabi, FloatAbi)?;
        key!(rustc_abi, RustcAbi)?;
        key!(relax_elf_relocations, bool);
        key!(llvm_args, list);
        key!(use_ctors_section, bool);
        key!(eh_frame_header, bool);
        key!(has_thumb_interworking, bool);
        key!(debuginfo_kind, DebuginfoKind)?;
        key!(split_debuginfo, SplitDebuginfo)?;
        key!(supported_split_debuginfo, fallible_list)?;
        key!(supported_sanitizers, SanitizerSet)?;
        key!(generate_arange_section, bool);
        key!(supports_stack_protector, bool);
        key!(small_data_threshold_support, SmallDataThresholdSupport)?;
        key!(entry_name);
        key!(supports_xray, bool);

        // we're going to run `update_from_cli`, but that won't change the target's AbiMap
        // FIXME: better factor the Target definition so we enforce this on a type level
        let abi_map = AbiMap::from_target(&base);

        if let Some(abi_str) = obj.remove("entry-abi") {
            if let Json::String(abi_str) = abi_str {
                match abi_str.parse::<ExternAbi>() {
                    Ok(abi) => base.options.entry_abi = abi_map.canonize_abi(abi, false).unwrap(),
                    Err(_) => return Err(format!("{abi_str} is not a valid ExternAbi")),
                }
            } else {
                incorrect_type.push("entry-abi".to_owned())
            }
        }

        base.update_from_cli();
        base.check_consistency(TargetKind::Json)?;

        // Each field should have been read using `Json::remove` so any keys remaining are unused.
        let remaining_keys = obj.keys();
        Ok((
            base,
            TargetWarnings { unused_fields: remaining_keys.cloned().collect(), incorrect_type },
        ))
    }
}

impl ToJson for Target {
    fn to_json(&self) -> Json {
        let mut d = serde_json::Map::new();
        let default: TargetOptions = Default::default();
        let mut target = self.clone();
        target.update_to_cli();

        macro_rules! target_val {
            ($attr:ident) => {{
                let name = (stringify!($attr)).replace("_", "-");
                d.insert(name, target.$attr.to_json());
            }};
        }

        macro_rules! target_option_val {
            ($attr:ident) => {{
                let name = (stringify!($attr)).replace("_", "-");
                if default.$attr != target.$attr {
                    d.insert(name, target.$attr.to_json());
                }
            }};
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
        d.insert("target-pointer-width".to_string(), self.pointer_width.to_string().to_json());
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
        target_option_val!(bitcode_llvm_cmdline);
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
