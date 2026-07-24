use std::ffi::{CString, c_char};

use rustc_ast::expand::typetree::{FncTree, Kind, TypeTree as RustTypeTree};

use crate::attributes;
use crate::context::FullCx;
use crate::llvm::{self, EnzymeWrapper, Value};

fn to_enzyme_typetree(
    rust_typetree: &RustTypeTree,
    _data_layout: &str,
    llcx: &llvm::Context,
) -> (llvm::TypeTree, Vec<llvm::TypeTree>) {
    let mut enzyme_tt = llvm::TypeTree::new();
    let extra_ints = process_typetree_recursive(&mut enzyme_tt, &rust_typetree, &[], llcx);

    let mut int_vec = vec![];
    for _ in 0..extra_ints {
        let mut int_tt = llvm::TypeTree::new();
        int_tt.insert(&[0], llvm::CConcreteType::DT_Integer, llcx);
        int_vec.push(int_tt);
    }

    (enzyme_tt, int_vec)
}

fn process_typetree_recursive(
    enzyme_tt: &mut llvm::TypeTree,
    rust_typetree: &RustTypeTree,
    parent_indices: &[i64],
    llcx: &llvm::Context,
) -> u32 {
    let mut extra_ints = 0;
    for rust_type in &rust_typetree.0 {
        let concrete_type = match rust_type.kind {
            Kind::Anything => llvm::CConcreteType::DT_Anything,
            Kind::Integer => llvm::CConcreteType::DT_Integer,
            Kind::Pointer => llvm::CConcreteType::DT_Pointer,
            Kind::RustSlice => llvm::CConcreteType::DT_Pointer,
            Kind::Half => llvm::CConcreteType::DT_Half,
            Kind::Float => llvm::CConcreteType::DT_Float,
            Kind::Double => llvm::CConcreteType::DT_Double,
            Kind::F128 => llvm::CConcreteType::DT_FP128,
            Kind::Unknown => llvm::CConcreteType::DT_Unknown,
        };

        let mut indices = parent_indices.to_vec();
        if !parent_indices.is_empty() {
            indices.push(rust_type.offset as i64);
        } else if rust_type.offset == -1 {
            indices.push(-1);
        } else {
            indices.push(rust_type.offset as i64);
        }

        enzyme_tt.insert(&indices, concrete_type, llcx);

        if matches!(rust_type.kind, Kind::RustSlice) {
            // We lower slices to `ptr,int`, so add the int here.
            extra_ints += 1;
        }

        if matches!(rust_type.kind, Kind::Pointer | Kind::RustSlice)
            && !rust_type.child.0.is_empty()
        {
            process_typetree_recursive(enzyme_tt, &rust_type.child, &indices, llcx);
        }
    }
    extra_ints
}

// Describes all the locations in which we know how to apply an Enzyme TypeTree.
enum TTLocation {
    Definition,
    Callsite,
    Intrinsic,
}

#[cfg_attr(not(feature = "llvm_enzyme"), allow(unused))]
pub(crate) fn add_tt<'tcx, 'll>(cx: &FullCx<'ll, 'tcx>, fn_def: &'ll Value, tt: FncTree) {
    // TypeTree processing uses functions from Enzyme, which we might not have available if we did
    // not build this compiler with `llvm_enzyme`. This feature is not strictly necessary, but
    // skipping this function increases the chance that Enzyme fails to compile some code.
    // FIXME(autodiff): In the future we should conditionally run this function even without the
    // `llvm_enzyme` feature, in case that libEnzyme was provided via rustup.
    #[cfg(not(feature = "llvm_enzyme"))]
    return;

    let tcx = cx.tcx;
    if !tcx.sess.opts.unstable_opts.autodiff.contains(&rustc_session::config::AutoDiff::Enable) {
        return;
    }
    if tcx.sess.opts.unstable_opts.autodiff.contains(&rustc_session::config::AutoDiff::NoTT) {
        return;
    }

    let llmod = cx.llmod;
    let llcx = cx.llcx;
    let inputs = tt.args;
    let ret_tt: RustTypeTree = tt.ret;

    let llvm_data_layout: *const c_char = unsafe { llvm::LLVMGetDataLayoutStr(&*llmod) };
    let llvm_data_layout =
        std::str::from_utf8(unsafe { std::ffi::CStr::from_ptr(llvm_data_layout) }.to_bytes())
            .expect("got a non-UTF8 data-layout from LLVM");

    let attr_name = "enzyme_type";
    let c_attr_name = CString::new(attr_name).unwrap();

    let tt_location: TTLocation = if llvm::LLVMRustIsCall(fn_def) {
        TTLocation::Callsite
    } else if llvm::LLVMRustSupportsEnzymeMD(fn_def) {
        TTLocation::Intrinsic
    } else {
        TTLocation::Definition
    };

    let mut offset = 0;
    for (i, input) in inputs.iter().enumerate() {
        let (enzyme_tt, extra_ints) = to_enzyme_typetree(&input, llvm_data_layout, llcx);

        // This scope is a simple solution since we *must* drop the enzyme_wrapper before
        // we drop any typetrees (mainly enzyme_tt and extra_ints). Drop calls can not accept
        // arguments like an enzyme_wrapper, so the typetree drop impl has to call get_instance
        // on the static enzyme instance, which is behind a Mutex. Therefore we'd deadlock if we
        // hold the enzyme_wrapper while dropping the typetrees.
        {
            let enzyme_wrapper = EnzymeWrapper::get_instance();
            let c_str = enzyme_wrapper.tree_to_cstr(enzyme_tt.inner);

            let attr = llvm::CreateAttrStringValueFromCStr(llcx, &c_attr_name, &c_str);
            let arg_pos = llvm::AttributePlace::Argument(i as u32 + offset);
            // FIXME(autodiff): We currently know that this is correct for all the cases in which we
            // call this function. But we should make it more robust for the future.
            match tt_location {
                TTLocation::Definition => {
                    attributes::apply_to_llfn(fn_def, arg_pos, &[attr]);
                }
                TTLocation::Callsite => {
                    attributes::apply_to_callsite(fn_def, arg_pos, &[attr]);
                }
                TTLocation::Intrinsic => {
                    let md = enzyme_wrapper.tree_to_md(enzyme_tt.inner, llcx);
                    unsafe {
                        llvm::LLVMRustSetEnzymeTypeMD(fn_def, md.unwrap());
                    }
                }
            }
            enzyme_wrapper.tree_to_string_free(c_str.as_ptr());
            for v in &extra_ints {
                offset += 1;
                let c_str = enzyme_wrapper.tree_to_cstr(v.inner);
                let int_attr = llvm::CreateAttrStringValueFromCStr(llcx, &c_attr_name, &c_str);
                let arg_pos = llvm::AttributePlace::Argument(i as u32 + offset);
                match tt_location {
                    TTLocation::Intrinsic => {
                        let md = enzyme_wrapper.tree_to_md(enzyme_tt.inner, llcx);
                        unsafe {
                            llvm::LLVMRustSetEnzymeTypeMD(fn_def, md.unwrap());
                        }
                    }
                    TTLocation::Definition => {
                        attributes::apply_to_llfn(fn_def, arg_pos, &[int_attr]);
                    }
                    TTLocation::Callsite => {
                        attributes::apply_to_callsite(fn_def, arg_pos, &[int_attr]);
                    }
                }
                enzyme_wrapper.tree_to_string_free(c_str.as_ptr());
            }
        }
    }
    // We will fail here if Rust types got lowered to LLVM in a way that we didn't predict.
    // We Error, so we can learn from our mistakes.
    if matches!(tt_location, TTLocation::Definition) {
        let expected = offset as usize + inputs.len();
        let actual = llvm::count_params(fn_def) as usize;
        if expected != actual {
            tcx.dcx().warn(format!(
                "autodiff type-tree failure. We expected {expected} LLVM argument(s), \
                 but the generated LLVM function has {actual} parameter(s)"
            ));
        }
    }

    // FIXME(autodiff): We should think more about what it means if a function returns a slice or
    // other fat ptrs.
    let (enzyme_tt, _extra_ints) = to_enzyme_typetree(&ret_tt, llvm_data_layout, llcx);
    if ret_tt != RustTypeTree::new() {
        let enzyme_wrapper = EnzymeWrapper::get_instance();
        let c_str = enzyme_wrapper.tree_to_cstr(enzyme_tt.inner);
        let ret_attr = llvm::CreateAttrStringValueFromCStr(llcx, &c_attr_name, &c_str);
        let arg_pos = llvm::AttributePlace::ReturnValue;
        match tt_location {
            TTLocation::Definition => {
                attributes::apply_to_llfn(fn_def, arg_pos, &[ret_attr]);
            }
            TTLocation::Callsite => {
                attributes::apply_to_callsite(fn_def, arg_pos, &[ret_attr]);
            }
            TTLocation::Intrinsic => {
                let md = enzyme_wrapper.tree_to_md(enzyme_tt.inner, llcx);
                unsafe {
                    llvm::LLVMRustSetEnzymeTypeMD(fn_def, md.unwrap());
                }
            }
        }
        enzyme_wrapper.tree_to_string_free(c_str.as_ptr());
    }
}
