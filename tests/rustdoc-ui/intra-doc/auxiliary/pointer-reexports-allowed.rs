#![feature(intra_doc_pointers)]
#![crate_name = "inner"]
/// Link to [some pointer](*const::to_raw_parts)
pub fn foo() {}
