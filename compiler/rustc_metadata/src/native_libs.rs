use rustc_abi::ExternAbi;
use rustc_attr_parsing::eval_config_entry;
use rustc_crate_store::{
    DllCallingConvention, DllImport, DllImportSymbolType, ForeignModule, NativeLib,
};
use rustc_data_structures::fx::FxHashSet;
use rustc_hir::attrs::PeImportNameType;
use rustc_hir::def::DefKind;
use rustc_hir::find_attr;
use rustc_middle::bug;
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrFlags;
use rustc_middle::query::LocalCrate;
use rustc_middle::ty::{self, List, Ty, TyCtxt};
use rustc_session::Session;
use rustc_span::Symbol;
use rustc_span::def_id::{DefId, LOCAL_CRATE};
use rustc_structures::NativeLibKind;
use rustc_target::spec::{Arch, BinaryFormat, CfgAbi};

use crate::diagnostics;

pub(crate) fn collect(tcx: TyCtxt<'_>, LocalCrate: LocalCrate) -> Vec<NativeLib> {
    let mut collector = Collector { tcx, libs: Vec::new() };
    if tcx.sess.opts.unstable_opts.link_directives {
        for module in tcx.foreign_modules(LOCAL_CRATE).values() {
            collector.process_module(module);
        }
    }
    collector.process_command_line();
    for lib in &mut collector.libs {
        // FIXME(jchlanda) Pauthtest does not support static linking. It must be dynamically linked,
        // with a dynamic linker acting as the ELF interpreter that can resolve pauth relocations
        // and enforce pointer authentication constraints.
        if tcx.sess.target.cfg_abi == CfgAbi::Pauthtest {
            if let NativeLibKind::Static { .. } = lib.kind {
                if !tcx.sess.opts.unstable_opts.ui_testing {
                    let diag = if lib.foreign_module.is_none() {
                        diagnostics::StaticLinkingNotSupported::UserRequested {
                            lib_name: lib.name,
                            target: tcx.sess.target.llvm_target.as_ref(),
                        }
                    } else {
                        diagnostics::StaticLinkingNotSupported::FromDependency {
                            lib_name: lib.name,
                            target: tcx.sess.target.llvm_target.as_ref(),
                        }
                    };
                    tcx.dcx().emit_warn(diag);
                }

                lib.kind = NativeLibKind::Dylib { as_needed: None };
            }
        }
    }
    collector.libs
}

pub(crate) fn relevant_lib(sess: &Session, lib: &NativeLib) -> bool {
    match lib.cfg {
        Some(ref cfg) => eval_config_entry(sess, cfg).as_bool(),
        None => true,
    }
}

struct Collector<'tcx> {
    tcx: TyCtxt<'tcx>,
    libs: Vec<NativeLib>,
}

impl<'tcx> Collector<'tcx> {
    fn process_module(&mut self, module: &ForeignModule) {
        let ForeignModule { def_id, abi, ref foreign_items } = *module;
        let def_id = def_id.expect_local();

        let sess = self.tcx.sess;

        if matches!(abi, ExternAbi::Rust) {
            return;
        }

        for attr in find_attr!(self.tcx, def_id, Link(links, _) => links).into_flat_iter() {
            let dll_imports = match attr.kind {
                NativeLibKind::RawDylib { .. } => foreign_items
                    .iter()
                    .filter_map(|&child_item| {
                        self.build_dll_import(
                            abi,
                            attr.import_name_type.map(|(import_name_type, _)| import_name_type),
                            child_item,
                        )
                    })
                    .collect(),
                _ => {
                    for &child_item in foreign_items {
                        if let Some(span) =
                            find_attr!(self.tcx, child_item, LinkOrdinal {span, ..} => *span)
                        {
                            sess.dcx().emit_err(diagnostics::LinkOrdinalRawDylib { span });
                        }
                    }

                    Vec::new()
                }
            };

            self.libs.push(NativeLib {
                name: attr.name,
                kind: attr.kind,
                cfg: attr.cfg.clone(),
                foreign_module: Some(def_id.to_def_id()),
                verbatim: attr.verbatim,
                dll_imports,
            });
        }
    }

    // Process libs passed on the command line
    fn process_command_line(&mut self) {
        // First, check for errors
        let mut renames = FxHashSet::default();
        for lib in &self.tcx.sess.opts.libs {
            if let NativeLibKind::Framework { .. } = lib.kind
                && !self.tcx.sess.target.is_like_darwin
            {
                // Cannot check this when parsing options because the target is not yet available.
                self.tcx.dcx().emit_err(diagnostics::LibFrameworkApple);
            }
            if let Some(ref new_name) = lib.new_name {
                let any_duplicate = self.libs.iter().any(|n| n.name.as_str() == lib.name);
                if new_name.is_empty() {
                    self.tcx
                        .dcx()
                        .emit_err(diagnostics::EmptyRenamingTarget { lib_name: &lib.name });
                } else if !any_duplicate {
                    self.tcx.dcx().emit_err(diagnostics::RenamingNoLink { lib_name: &lib.name });
                } else if !renames.insert(&lib.name) {
                    self.tcx.dcx().emit_err(diagnostics::MultipleRenamings { lib_name: &lib.name });
                }
            }
        }

        // Update kind and, optionally, the name of all native libraries
        // (there may be more than one) with the specified name. If any
        // library is mentioned more than once, keep the latest mention
        // of it, so that any possible dependent libraries appear before
        // it. (This ensures that the linker is able to see symbols from
        // all possible dependent libraries before linking in the library
        // in question.)
        for passed_lib in &self.tcx.sess.opts.libs {
            // If we've already added any native libraries with the same
            // name, they will be pulled out into `existing`, so that we
            // can move them to the end of the list below.
            let mut existing = self
                .libs
                .extract_if(.., |lib| {
                    if lib.name.as_str() == passed_lib.name {
                        // FIXME: This whole logic is questionable, whether modifiers are
                        // involved or not, library reordering and kind overriding without
                        // explicit `:rename` in particular.
                        if lib.has_modifiers() || passed_lib.has_modifiers() {
                            match lib.foreign_module {
                                Some(def_id) => {
                                    self.tcx.dcx().emit_err(diagnostics::NoLinkModOverride {
                                        span: Some(self.tcx.def_span(def_id)),
                                    })
                                }
                                None => self
                                    .tcx
                                    .dcx()
                                    .emit_err(diagnostics::NoLinkModOverride { span: None }),
                            };
                        }
                        if passed_lib.kind != NativeLibKind::Unspecified {
                            lib.kind = passed_lib.kind;
                        }
                        if let Some(new_name) = &passed_lib.new_name {
                            lib.name = Symbol::intern(new_name);
                        }
                        lib.verbatim = passed_lib.verbatim;
                        return true;
                    }
                    false
                })
                .collect::<Vec<_>>();
            if existing.is_empty() {
                // Add if not found
                let new_name: Option<&str> = passed_lib.new_name.as_deref();
                let name = Symbol::intern(new_name.unwrap_or(&passed_lib.name));
                self.libs.push(NativeLib {
                    name,
                    kind: passed_lib.kind,
                    cfg: None,
                    foreign_module: None,
                    verbatim: passed_lib.verbatim,
                    dll_imports: Vec::new(),
                });
            } else {
                // Move all existing libraries with the same name to the
                // end of the command line.
                self.libs.append(&mut existing);
            }
        }
    }

    fn i686_arg_list_size(&self, item: DefId) -> usize {
        let argument_types: &List<Ty<'_>> = self.tcx.instantiate_bound_regions_with_erased(
            self.tcx
                .type_of(item)
                .instantiate_identity()
                .skip_norm_wip()
                .fn_sig(self.tcx)
                .inputs()
                .map_bound(|slice| self.tcx.mk_type_list(slice)),
        );

        argument_types
            .iter()
            .map(|ty| {
                let layout = self
                    .tcx
                    .layout_of(ty::TypingEnv::fully_monomorphized().as_query_input(ty))
                    .expect("layout")
                    .layout;
                // In both stdcall and fastcall, we always round up the argument size to the
                // nearest multiple of 4 bytes.
                (layout.size().bytes_usize() + 3) & !3
            })
            .sum()
    }

    fn build_dll_import(
        &self,
        abi: ExternAbi,
        import_name_type: Option<PeImportNameType>,
        item: DefId,
    ) -> Option<DllImport> {
        let span = self.tcx.def_span(item);

        // This `extern` block should have been checked for general ABI support before, but let's
        // double-check that.
        assert!(self.tcx.sess.target.is_abi_supported(abi));

        // This logic is similar to `AbiMap::canonize_abi` (in rustc_target/src/spec/abi_map.rs) but
        // we need more detail than those adjustments, and we can't support all ABIs that are
        // generally supported.
        let calling_convention = if self.tcx.sess.target.arch == Arch::X86 {
            match abi {
                ExternAbi::C { .. } | ExternAbi::Cdecl { .. } => DllCallingConvention::C,
                ExternAbi::Stdcall { .. } => {
                    DllCallingConvention::Stdcall(self.i686_arg_list_size(item))
                }
                // On Windows, `extern "system"` behaves like msvc's `__stdcall`.
                // `__stdcall` only applies on x86 and on non-variadic functions:
                // https://learn.microsoft.com/en-us/cpp/cpp/stdcall?view=msvc-170
                ExternAbi::System { .. } => {
                    let c_variadic = self
                        .tcx
                        .type_of(item)
                        .instantiate_identity()
                        .skip_norm_wip()
                        .fn_sig(self.tcx)
                        .c_variadic();

                    if c_variadic {
                        DllCallingConvention::C
                    } else {
                        DllCallingConvention::Stdcall(self.i686_arg_list_size(item))
                    }
                }
                ExternAbi::Fastcall { .. } => {
                    DllCallingConvention::Fastcall(self.i686_arg_list_size(item))
                }
                ExternAbi::Vectorcall { .. } => {
                    DllCallingConvention::Vectorcall(self.i686_arg_list_size(item))
                }
                _ => {
                    self.tcx.dcx().emit_err(diagnostics::RawDylibUnsupportedAbi { span });
                    return None;
                }
            }
        } else {
            match abi {
                ExternAbi::C { .. } | ExternAbi::Win64 { .. } | ExternAbi::System { .. } => {
                    DllCallingConvention::C
                }
                _ => {
                    self.tcx.dcx().emit_err(diagnostics::RawDylibUnsupportedAbi { span });
                    return None;
                }
            }
        };

        let codegen_fn_attrs = self.tcx.codegen_fn_attrs(item);
        let import_name_type = codegen_fn_attrs
            .link_ordinal
            .map_or(import_name_type, |ord| Some(PeImportNameType::Ordinal(ord)));

        let name = codegen_fn_attrs.symbol_name.unwrap_or_else(|| self.tcx.item_name(item));

        if self.tcx.sess.target.binary_format == BinaryFormat::Elf {
            let name = name.as_str();
            if name.contains('\0') {
                self.tcx.dcx().emit_err(diagnostics::RawDylibMalformed { span });
            } else if let Some((left, right)) = name.split_once('@')
                && (left.is_empty() || right.is_empty() || right.contains('@'))
            {
                self.tcx.dcx().emit_err(diagnostics::RawDylibMalformed { span });
            }
        }

        let def_kind = self.tcx.def_kind(item);
        let symbol_type = if def_kind.is_fn_like() {
            DllImportSymbolType::Function
        } else if matches!(def_kind, DefKind::Static { .. }) {
            if codegen_fn_attrs.flags.contains(CodegenFnAttrFlags::THREAD_LOCAL) {
                DllImportSymbolType::ThreadLocal
            } else {
                DllImportSymbolType::Static
            }
        } else if def_kind == DefKind::ForeignTy {
            return None;
        } else {
            bug!("Unexpected type for raw-dylib: {}", def_kind.descr(item));
        };

        let size = match symbol_type {
            // We cannot determine the size of a function at compile time, but it shouldn't matter anyway.
            DllImportSymbolType::Function => rustc_abi::Size::ZERO,
            DllImportSymbolType::Static | DllImportSymbolType::ThreadLocal => {
                let ty = self.tcx.type_of(item).instantiate_identity().skip_norm_wip();
                self.tcx
                    .layout_of(ty::TypingEnv::fully_monomorphized().as_query_input(ty))
                    .ok()
                    .map(|layout| layout.size)
                    .unwrap_or_else(|| bug!("Non-function symbols must have a size"))
            }
        };

        Some(DllImport { name, import_name_type, calling_convention, span, symbol_type, size })
    }
}
