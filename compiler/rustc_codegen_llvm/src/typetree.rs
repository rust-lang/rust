use std::ffi::{CString, c_char, c_uint};

use rustc_ast::expand::typetree::{FncTree, TypeTree as RustTypeTree};

use crate::attributes;
use crate::llvm::{self, Value};

/// Converts a Rust TypeTree to Enzyme's internal TypeTree format
///
/// This function takes a Rust-side TypeTree (from rustc_ast::expand::typetree)
/// and converts it to Enzyme's internal C++ TypeTree representation that
/// Enzyme can understand during differentiation analysis.
#[cfg(llvm_enzyme)]
fn to_enzyme_typetree(
    rust_typetree: RustTypeTree,
    data_layout: &str,
    llcx: &llvm::Context,
) -> llvm::TypeTree {
    // Start with an empty TypeTree
    let mut enzyme_tt = llvm::TypeTree::new();

    // Convert each Type in the Rust TypeTree to Enzyme format
    for rust_type in rust_typetree.0 {
        let concrete_type = match rust_type.kind {
            rustc_ast::expand::typetree::Kind::Anything => llvm::CConcreteType::DT_Anything,
            rustc_ast::expand::typetree::Kind::Integer => llvm::CConcreteType::DT_Integer,
            rustc_ast::expand::typetree::Kind::Pointer => llvm::CConcreteType::DT_Pointer,
            rustc_ast::expand::typetree::Kind::Half => llvm::CConcreteType::DT_Half,
            rustc_ast::expand::typetree::Kind::Float => llvm::CConcreteType::DT_Float,
            rustc_ast::expand::typetree::Kind::Double => llvm::CConcreteType::DT_Double,
            rustc_ast::expand::typetree::Kind::F128 => llvm::CConcreteType::DT_Unknown,
            rustc_ast::expand::typetree::Kind::Unknown => llvm::CConcreteType::DT_Unknown,
        };

        // Create a TypeTree for this specific type
        let type_tt = llvm::TypeTree::from_type(concrete_type, llcx);

        // Apply offset if specified
        let type_tt = if rust_type.offset == -1 {
            type_tt // -1 means everywhere/no specific offset
        } else {
            // Apply specific offset positioning
            type_tt.shift(data_layout, rust_type.offset, rust_type.size as isize, 0)
        };

        // Merge this type into the main TypeTree
        enzyme_tt = enzyme_tt.merge(type_tt);
    }

    enzyme_tt
}

#[cfg(not(llvm_enzyme))]
fn to_enzyme_typetree(
    _rust_typetree: RustTypeTree,
    _data_layout: &str,
    _llcx: &llvm::Context,
) -> ! {
    unimplemented!("TypeTree conversion not available without llvm_enzyme support")
}

// Attaches TypeTree information to LLVM function as enzyme_type attributes.
#[cfg(llvm_enzyme)]
pub(crate) fn add_tt<'ll>(
    llmod: &'ll llvm::Module,
    llcx: &'ll llvm::Context,
    fn_def: &'ll Value,
    tt: FncTree,
) {
    let inputs = tt.args;
    let ret_tt: RustTypeTree = tt.ret;

    // Get LLVM data layout string for TypeTree conversion
    let llvm_data_layout: *const c_char = unsafe { llvm::LLVMGetDataLayoutStr(&*llmod) };
    let llvm_data_layout =
        std::str::from_utf8(unsafe { std::ffi::CStr::from_ptr(llvm_data_layout) }.to_bytes())
            .expect("got a non-UTF8 data-layout from LLVM");

    // Attribute name that Enzyme recognizes for TypeTree information
    let attr_name = "enzyme_type";
    let c_attr_name = CString::new(attr_name).unwrap();

    // Attach TypeTree attributes to each input parameter
    // Enzyme uses these to understand parameter memory layouts during differentiation
    for (i, input) in inputs.iter().enumerate() {
        unsafe {
            // Convert Rust TypeTree to Enzyme's internal format
            let enzyme_tt = to_enzyme_typetree(input.clone(), llvm_data_layout, llcx);

            // Serialize TypeTree to string format that Enzyme can parse
            let c_str = llvm::EnzymeTypeTreeToString(enzyme_tt.inner);
            let c_str = std::ffi::CStr::from_ptr(c_str);

            // Create LLVM string attribute with TypeTree information
            let attr = llvm::LLVMCreateStringAttribute(
                llcx,
                c_attr_name.as_ptr(),
                c_attr_name.as_bytes().len() as c_uint,
                c_str.as_ptr(),
                c_str.to_bytes().len() as c_uint,
            );

            // Attach attribute to the specific function parameter
            // Note: ArgumentPlace uses 0-based indexing, but LLVM uses 1-based for arguments
            attributes::apply_to_llfn(fn_def, llvm::AttributePlace::Argument(i as u32), &[attr]);

            // Free the C string to prevent memory leaks
            llvm::EnzymeTypeTreeToStringFree(c_str.as_ptr());
        }
    }

    // Attach TypeTree attribute to the return type
    // Enzyme needs this to understand how to handle return value derivatives
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

        // Attach to function return type
        attributes::apply_to_llfn(fn_def, llvm::AttributePlace::ReturnValue, &[ret_attr]);

        // Free the C string
        llvm::EnzymeTypeTreeToStringFree(c_str.as_ptr());
    }
}

// Fallback implementation when Enzyme is not available
#[cfg(not(llvm_enzyme))]
pub(crate) fn add_tt<'ll>(
    _llmod: &'ll llvm::Module,
    _llcx: &'ll llvm::Context,
    _fn_def: &'ll Value,
    _tt: FncTree,
) {
    unimplemented!()
}
