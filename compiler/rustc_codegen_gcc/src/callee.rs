#[cfg(feature = "master")]
use gccjit::{FnAttribute, Visibility};
use gccjit::{Function, FunctionType};
use rustc_middle::ty::layout::{FnAbiOf, HasTyCtxt};
use rustc_middle::ty::{self, Instance, TypeVisitableExt};

use crate::attributes;
use crate::context::CodegenCx;

/// Codegens a reference to a fn/method item, monomorphizing and
/// inlining as it goes.
///
/// # Parameters
///
/// - `cx`: the crate context
/// - `instance`: the instance to be instantiated
pub fn get_fn<'gcc, 'tcx>(cx: &CodegenCx<'gcc, 'tcx>, instance: Instance<'tcx>) -> Function<'gcc> {
    let tcx = cx.tcx();

    assert!(!instance.args.has_infer());
    assert!(!instance.args.has_escaping_bound_vars());

    let sym = tcx.symbol_name(instance).name;

    if let Some(&func) = cx.function_instances.borrow().get(&instance) {
        return func;
    }

    let fn_abi = cx.fn_abi_of_instance(instance, ty::List::empty());

    let func = if let Some(_func) = cx.get_declared_value(sym) {
        // FIXME(antoyo): we never reach this because get_declared_value only returns global variables
        // and here we try to get a function.
        unreachable!();
        /*
        // Create a fn pointer with the new signature.
        let ptrtype = fn_abi.ptr_to_gcc_type(cx);

        // This is subtle and surprising, but sometimes we have to bitcast
        // the resulting fn pointer.  The reason has to do with external
        // functions.  If you have two crates that both bind the same C
        // library, they may not use precisely the same types: for
        // example, they will probably each declare their own structs,
        // which are distinct types from LLVM's point of view (nominal
        // types).
        //
        // Now, if those two crates are linked into an application, and
        // they contain inlined code, you can wind up with a situation
        // where both of those functions wind up being loaded into this
        // application simultaneously. In that case, the same function
        // (from LLVM's point of view) requires two types. But of course
        // LLVM won't allow one function to have two types.
        //
        // What we currently do, therefore, is declare the function with
        // one of the two types (whichever happens to come first) and then
        // bitcast as needed when the function is referenced to make sure
        // it has the type we expect.
        //
        // This can occur on either a crate-local or crate-external
        // reference. It also occurs when testing libcore and in some
        // other weird situations. Annoying.
        if cx.val_ty(func) != ptrtype {
            // TODO(antoyo): cast the pointer.
            func
        }
        else {
            func
        }*/
    } else {
        cx.linkage.set(FunctionType::Extern);
        let func = cx.declare_fn(sym, fn_abi);

        attributes::from_fn_attrs(cx, func, instance);

        #[cfg(feature = "master")]
        {
            let instance_def_id = instance.def_id();

            // TODO(antoyo): set linkage and attributes.

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
            let is_generic = instance.args.non_erasable_generics().next().is_some();

            let is_hidden = if is_generic {
                // This is a monomorphization of a generic function.
                if !(cx.tcx.sess.opts.share_generics()
                    || tcx.codegen_fn_attrs(instance_def_id).inline
                        == rustc_attr_data_structures::InlineAttr::Never)
                {
                    // When not sharing generics, all instances are in the same
                    // crate and have hidden visibility.
                    true
                } else if let Some(instance_def_id) = instance_def_id.as_local() {
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
                func.add_attribute(FnAttribute::Visibility(Visibility::Hidden));
            }
        }

        func
    };

    cx.function_instances.borrow_mut().insert(instance, func);

    func
}
