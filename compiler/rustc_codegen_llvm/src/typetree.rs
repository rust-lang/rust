use rustc_ast::expand::typetree::FncTree;
#[cfg(feature = "llvm_enzyme")]
use {
    crate::attributes,
    rustc_ast::expand::typetree::TypeTree as RustTypeTree,
    std::ffi::{CString, c_char, c_uint},
};

use crate::llvm::{self, Value};

#[cfg(feature = "llvm_enzyme")]
fn to_enzyme_typetree(
    rust_typetree: RustTypeTree,
    _data_layout: &str,
    llcx: &llvm::Context,
) -> llvm::TypeTree {
    let mut enzyme_tt = llvm::TypeTree::new();
    process_typetree_recursive(&mut enzyme_tt, &rust_typetree, &[], llcx);
    enzyme_tt
}
#[cfg(feature = "llvm_enzyme")]
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

#[cfg(feature = "llvm_enzyme")]
pub(crate) fn add_tt<'ll>(
    llmod: &'ll llvm::Module,
    llcx: &'ll llvm::Context,
    fn_def: &'ll Value,
    tt: FncTree,
) {
    let inputs = tt.args;
    let ret_tt: RustTypeTree = tt.ret;

    let llvm_data_layout: *const c_char = unsafe { llvm::LLVMGetDataLayoutStr(&*llmod) };
    let llvm_data_layout =
        std::str::from_utf8(unsafe { std::ffi::CStr::from_ptr(llvm_data_layout) }.to_bytes())
            .expect("got a non-UTF8 data-layout from LLVM");

    let attr_name = "enzyme_type";
    let c_attr_name = CString::new(attr_name).unwrap();

    for (i, input) in inputs.iter().enumerate() {
        unsafe {
            let enzyme_tt = to_enzyme_typetree(input.clone(), llvm_data_layout, llcx);
            let c_str = llvm::EnzymeTypeTreeToString(enzyme_tt.inner);
            let c_str = std::ffi::CStr::from_ptr(c_str);

            let attr = llvm::LLVMCreateStringAttribute(
                llcx,
                c_attr_name.as_ptr(),
                c_attr_name.as_bytes().len() as c_uint,
                c_str.as_ptr(),
                c_str.to_bytes().len() as c_uint,
            );

            attributes::apply_to_llfn(fn_def, llvm::AttributePlace::Argument(i as u32), &[attr]);
            llvm::EnzymeTypeTreeToStringFree(c_str.as_ptr());
        }
    }

    unsafe {
        let enzyme_tt = to_enzyme_typetree(ret_tt, llvm_data_layout, llcx);
        let c_str = llvm::EnzymeTypeTreeToString(enzyme_tt.inner);
        let c_str = std::ffi::CStr::from_ptr(c_str);

        let ret_attr = llvm::LLVMCreateStringAttribute(
            llcx,
            c_attr_name.as_ptr(),
            c_attr_name.as_bytes().len() as c_uint,
            c_str.as_ptr(),
            c_str.to_bytes().len() as c_uint,
        );

        attributes::apply_to_llfn(fn_def, llvm::AttributePlace::ReturnValue, &[ret_attr]);
        llvm::EnzymeTypeTreeToStringFree(c_str.as_ptr());
    }
}

#[cfg(not(feature = "llvm_enzyme"))]
pub(crate) fn add_tt<'ll>(
    _llmod: &'ll llvm::Module,
    _llcx: &'ll llvm::Context,
    _fn_def: &'ll Value,
    _tt: FncTree,
) {
    unimplemented!()
}
