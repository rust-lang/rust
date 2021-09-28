use gccjit::{FunctionType, RValue};
use rustc_codegen_ssa::traits::BaseTypeMethods;
use rustc_middle::ty::{self, Instance, TypeFoldable};
use rustc_middle::ty::layout::{FnAbiOf, HasTyCtxt};

use crate::abi::FnAbiGccExt;
use crate::context::CodegenCx;

/// Codegens a reference to a fn/method item, monomorphizing and
/// inlining as it goes.
///
/// # Parameters
///
/// - `cx`: the crate context
/// - `instance`: the instance to be instantiated
pub fn get_fn<'gcc, 'tcx>(cx: &CodegenCx<'gcc, 'tcx>, instance: Instance<'tcx>) -> RValue<'gcc> {
    let tcx = cx.tcx();

    assert!(!instance.substs.needs_infer());
    assert!(!instance.substs.has_escaping_bound_vars());

    if let Some(&func) = cx.function_instances.borrow().get(&instance) {
        return func;
    }

    let sym = tcx.symbol_name(instance).name;

    let fn_abi = cx.fn_abi_of_instance(instance, ty::List::empty());

    let func =
        if let Some(func) = cx.get_declared_value(&sym) {
            // Create a fn pointer with the new signature.
            let ptrty = fn_abi.ptr_to_gcc_type(cx);

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
            if cx.val_ty(func) != ptrty {
                // TODO(antoyo): cast the pointer.
                func
            }
            else {
                func
            }
        }
        else {
            cx.linkage.set(FunctionType::Extern);
            let func = cx.declare_fn(&sym, &fn_abi);

            // TODO(antoyo): set linkage and attributes.
            func
        };

    cx.function_instances.borrow_mut().insert(instance, func);

    func
}
