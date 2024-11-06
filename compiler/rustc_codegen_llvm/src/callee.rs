//! Handles codegen of callees as well as other call-related
//! things. Callees are a superset of normal rust values and sometimes
//! have different representations. In particular, top-level fn items
//! and methods are represented as just a fn ptr and not a full
//! closure.

use rustc_codegen_ssa::common;
use rustc_middle::ty::layout::{FnAbiOf, HasTyCtxt, HasTypingEnv};
use rustc_middle::ty::{self, Instance, TypeVisitableExt};
use tracing::debug;

use crate::context::CodegenCx;
use crate::llvm;
use crate::value::Value;

/// Codegens a reference to a fn/method item, monomorphizing and
/// inlining as it goes.
pub(crate) fn get_fn<'ll, 'tcx>(cx: &CodegenCx<'ll, 'tcx>, instance: Instance<'tcx>) -> &'ll Value {
    let tcx = cx.tcx();

    debug!("get_fn(instance={:?})", instance);

    assert!(!instance.args.has_infer());
    assert!(!instance.args.has_escaping_bound_vars());

    if let Some(&llfn) = cx.instances.borrow().get(&instance) {
        return llfn;
    }

    let sym = tcx.symbol_name(instance).name;
    debug!("get_fn({:?}: {:?}) => {}", instance, instance.ty(cx.tcx(), cx.typing_env()), sym);

    let fn_abi = cx.fn_abi_of_instance(instance, ty::List::empty());

    let llfn = if let Some(llfn) = cx.get_declared_value(sym) {
        llfn
    } else {
        let instance_def_id = instance.def_id();
        let llfn = if tcx.sess.target.arch == "x86"
            && let Some(dllimport) = crate::common::get_dllimport(tcx, instance_def_id, sym)
        {
            // When calling functions in generated import libraries, MSVC needs
            // the fully decorated name (as would have been in the declaring
            // object file), but MinGW wants the name as exported (as would be
            // in the def file) which may be missing decorations.
            let mingw_gnu_toolchain = common::is_mingw_gnu_toolchain(&tcx.sess.target);
            let llfn = cx.declare_fn(
                &common::i686_decorated_name(
                    dllimport,
                    mingw_gnu_toolchain,
                    true,
                    !mingw_gnu_toolchain,
                ),
                fn_abi,
                Some(instance),
            );

            // Fix for https://github.com/rust-lang/rust/issues/104453
            // On x86 Windows, LLVM uses 'L' as the prefix for any private
            // global symbols, so when we create an undecorated function symbol
            // that begins with an 'L' LLVM misinterprets that as a private
            // global symbol that it created and so fails the compilation at a
            // later stage since such a symbol must have a definition.
            //
            // To avoid this, we set the Storage Class to "DllImport" so that
            // LLVM will prefix the name with `__imp_`. Ideally, we'd like the
            // existing logic below to set the Storage Class, but it has an
            // exemption for MinGW for backwards compatibility.
            unsafe {
                llvm::LLVMSetDLLStorageClass(llfn, llvm::DLLStorageClass::DllImport);
            }
            llfn
        } else {
            cx.declare_fn(sym, fn_abi, Some(instance))
        };
        debug!("get_fn: not casting pointer!");

        // Apply an appropriate linkage/visibility value to our item that we
        // just declared.
        //
        // This is sort of subtle. Inside our codegen unit we started off
        // compilation by predefining all our own `MonoItem` instances. That
        // is, everything we're codegenning ourselves is already defined. That
        // means that anything we're actually codegenning in this codegen unit
        // will have hit the above branch in `get_declared_value`. As a result,
        // we're guaranteed here that we're declaring a symbol that won't get
        // defined, or in other words we're referencing a value from another
        // codegen unit or even another crate.
        //
        // So because this is a foreign value we blanket apply an external
        // linkage directive because it's coming from a different object file.
        // The visibility here is where it gets tricky. This symbol could be
        // referencing some foreign crate or foreign library (an `extern`
        // block) in which case we want to leave the default visibility. We may
        // also, though, have multiple codegen units. It could be a
        // monomorphization, in which case its expected visibility depends on
        // whether we are sharing generics or not. The important thing here is
        // that the visibility we apply to the declaration is the same one that
        // has been applied to the definition (wherever that definition may be).

        llvm::set_linkage(llfn, llvm::Linkage::ExternalLinkage);
        unsafe {
            let is_generic = instance.args.non_erasable_generics().next().is_some();

            let is_hidden = if is_generic {
                // This is a monomorphization of a generic function.
                if !(cx.tcx.sess.opts.share_generics()
                    || tcx.codegen_fn_attrs(instance_def_id).inline == rustc_hir::InlineAttr::Never)
                {
                    // When not sharing generics, all instances are in the same
                    // crate and have hidden visibility.
                    true
                } else {
                    if let Some(instance_def_id) = instance_def_id.as_local() {
                        // This is a monomorphization of a generic function
                        // defined in the current crate. It is hidden if:
                        // - the definition is unreachable for downstream
                        //   crates, or
                        // - the current crate does not re-export generics
                        //   (because the crate is a C library or executable)
                        cx.tcx.is_unreachable_local_definition(instance_def_id)
                            || !cx.tcx.local_crate_exports_generics()
                    } else {
                        // This is a monomorphization of a generic function
                        // defined in an upstream crate. It is hidden if:
                        // - it is instantiated in this crate, and
                        // - the current crate does not re-export generics
                        instance.upstream_monomorphization(tcx).is_none()
                            && !cx.tcx.local_crate_exports_generics()
                    }
                }
            } else {
                // This is a non-generic function. It is hidden if:
                // - it is instantiated in the local crate, and
                //   - it is defined an upstream crate (non-local), or
                //   - it is not reachable
                cx.tcx.is_codegened_item(instance_def_id)
                    && (!instance_def_id.is_local()
                        || !cx.tcx.is_reachable_non_generic(instance_def_id))
            };
            if is_hidden {
                llvm::set_visibility(llfn, llvm::Visibility::Hidden);
            }

            // MinGW: For backward compatibility we rely on the linker to decide whether it
            // should use dllimport for functions.
            if cx.use_dll_storage_attrs
                && let Some(library) = tcx.native_library(instance_def_id)
                && library.kind.is_dllimport()
                && !matches!(tcx.sess.target.env.as_ref(), "gnu" | "uclibc")
            {
                llvm::LLVMSetDLLStorageClass(llfn, llvm::DLLStorageClass::DllImport);
            }

            if cx.should_assume_dso_local(llfn, true) {
                llvm::LLVMRustSetDSOLocal(llfn, true);
            }
        }

        llfn
    };

    cx.instances.borrow_mut().insert(instance, llfn);

    llfn
}
