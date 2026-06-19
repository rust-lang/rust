use std::ffi::{CString, c_char, c_uint};

use rustc_ast::expand::typetree::{FncTree, TypeTree as RustTypeTree};
use crate::llvm::LLVMRustSetEnzymeTypeMetadata;
use crate::llvm::LLVMRustIsLoadOrExtractValue;
use crate::attributes;
use crate::llvm::{self, EnzymeWrapper, TypeTree, Value};

fn to_enzyme_typetree(
    rust_typetree: RustTypeTree,
    _data_layout: &str,
    llcx: &llvm::Context,
) -> llvm::TypeTree {
    let mut enzyme_tt = llvm::TypeTree::new();
    process_typetree_recursive(&mut enzyme_tt, &rust_typetree, &[], llcx);
    enzyme_tt
}
fn process_typetree_recursive(
    enzyme_tt: &mut llvm::TypeTree,
    rust_typetree: &RustTypeTree,
    parent_indices: &[i64],
    llcx: &llvm::Context,
) {
    for rust_type in &rust_typetree.0 {
        let concrete_type = match rust_type.kind {
            rustc_ast::expand::typetree::Kind::Anything => llvm::CConcreteType::DT_Anything,
            rustc_ast::expand::typetree::Kind::Integer => llvm::CConcreteType::DT_Integer,
            rustc_ast::expand::typetree::Kind::Pointer => llvm::CConcreteType::DT_Pointer,
            rustc_ast::expand::typetree::Kind::Half => llvm::CConcreteType::DT_Half,
            rustc_ast::expand::typetree::Kind::Float => llvm::CConcreteType::DT_Float,
            rustc_ast::expand::typetree::Kind::Double => llvm::CConcreteType::DT_Double,
            rustc_ast::expand::typetree::Kind::F128 => llvm::CConcreteType::DT_FP128,
            rustc_ast::expand::typetree::Kind::Unknown => llvm::CConcreteType::DT_Unknown,
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

        if rust_type.kind == rustc_ast::expand::typetree::Kind::Pointer
            && !rust_type.child.0.is_empty()
        {
            process_typetree_recursive(enzyme_tt, &rust_type.child, &indices, llcx);
        }
    }
}

#[cfg_attr(not(feature = "llvm_enzyme"), allow(unused))]
pub(crate) fn add_tt<'ll>(
    llmod: &'ll llvm::Module,
    llcx: &'ll llvm::Context,
    fn_def: &'ll Value,
    tcx: rustc_middle::ty::TyCtxt<'_>,
    tt: FncTree,
) {
    if !tcx.sess.opts.unstable_opts.autodiff.contains(&rustc_session::config::AutoDiff::Enable) {
        return;
    }
    if tcx.sess.opts.unstable_opts.autodiff.contains(&rustc_session::config::AutoDiff::NoTT) {
        return;
    }
    // TypeTree processing uses functions from Enzyme, which we might not have available if we did
    // not build this compiler with `llvm_enzyme`. This feature is not strictly necessary, but
    // skipping this function increases the chance that Enzyme fails to compile some code.
    // FIXME(autodiff): In the future we should conditionally run this function even without the
    // `llvm_enzyme` feature, in case that libEnzyme was provided via rustup.
    #[cfg(not(feature = "llvm_enzyme"))]
    return;

    let inputs = tt.args;
    let ret_tt: RustTypeTree = tt.ret;

    //dbg!("getting DataLayout");
    let llvm_data_layout: *const c_char = unsafe { llvm::LLVMGetDataLayoutStr(&*llmod) };
    let llvm_data_layout =
        std::str::from_utf8(unsafe { std::ffi::CStr::from_ptr(llvm_data_layout) }.to_bytes())
            .expect("got a non-UTF8 data-layout from LLVM");

    let attr_name = "enzyme_type";
    let c_attr_name = CString::new(attr_name).unwrap();
    //dbg!("going to iter over inputs");

    for (i, input) in inputs.iter().enumerate() {
        unsafe {
            if *input == rustc_ast::expand::typetree::TypeTree::new() {
                //dbg!("skipping empty input tt");
                continue;
            }
            let enzyme_tt = to_enzyme_typetree(input.clone(), llvm_data_layout, llcx);
            let enzyme_wrapper = EnzymeWrapper::get_instance();
            let c_str = enzyme_wrapper.tree_to_string(enzyme_tt.inner);
            let c_str = std::ffi::CStr::from_ptr(c_str);

            let attr = llvm::LLVMCreateStringAttribute(
                llcx,
                c_attr_name.as_ptr(),
                c_attr_name.as_bytes().len() as c_uint,
                c_str.as_ptr(),
                c_str.to_bytes().len() as c_uint,
            );
            //dbg!("adding attribute for argument {}", i);
            //dbg!("attribute string: {:?}", c_str);
            //dbg!(&fn_def);

            if llvm::LLVMRustIsIntrinsicCall(fn_def) {
                //dbg!("intrinsic");
                attributes::apply_to_callsite(fn_def, llvm::AttributePlace::Argument(i as u32), &[attr]);
            //} else if LLVMRustIsPtrLoad(fn_def) {
            } else if LLVMRustIsLoadOrExtractValue(fn_def) {
                //dbg!("skipping input args for instr");
            } else {
                //dbg!("fn call");
                attributes::apply_to_llfn(fn_def, llvm::AttributePlace::Argument(i as u32), &[attr]);
            }
            enzyme_wrapper.tree_to_string_free(c_str.as_ptr());
        }
    }
    //dbg!("finished to iter over inputs");

    unsafe {
        if ret_tt == rustc_ast::expand::typetree::TypeTree::new() {
            //dbg!("skipping empty return tt");
            return;
        }
        let enzyme_tt = to_enzyme_typetree(ret_tt, llvm_data_layout, llcx);
        let enzyme_wrapper = EnzymeWrapper::get_instance();
        let c_str = enzyme_wrapper.tree_to_string(enzyme_tt.inner);
        // just printing
        //let ptr = wrapper.tree_to_string(self.inner);
        let cstr = unsafe { std::ffi::CStr::from_ptr(c_str) };
        use std::io::Write as _;
        let mut stderr = std::io::stderr().lock();

        match cstr.to_str() {
            Ok(x) => {
                writeln!(stderr, "parsed: {:?}", x).ok();
            }
            Err(err) => {
                writeln!(stderr, "could not parse: {}", err).ok();
            }
        }

        // delete C string pointer
        //wrapper.tree_to_string_free(ptr);
        // done printing
        let c_str = std::ffi::CStr::from_ptr(c_str);

        let ret_attr = llvm::LLVMCreateStringAttribute(
            llcx,
            c_attr_name.as_ptr(),
            c_attr_name.as_bytes().len() as c_uint,
            c_str.as_ptr(),
            c_str.to_bytes().len() as c_uint,
        );

        dbg!(&fn_def);

        if llvm::LLVMRustIsIntrinsicCall(fn_def) {
            dbg!("intrinsic call");
            attributes::apply_to_callsite(fn_def, llvm::AttributePlace::ReturnValue, &[ret_attr]);
        //} else if LLVMRustIsPtrLoad(fn_def) {
        } else if LLVMRustIsLoadOrExtractValue(fn_def) {
            let val = enzyme_wrapper.tree_to_md(enzyme_tt.inner, llcx);
            LLVMRustSetEnzymeTypeMetadata(fn_def, val.unwrap());
        } else {
            dbg!("fn call");
            attributes::apply_to_llfn(fn_def, llvm::AttributePlace::ReturnValue, &[ret_attr]);
        }
        enzyme_wrapper.tree_to_string_free(c_str.as_ptr());
    }
    dbg!("finished to add return attribute");
}
