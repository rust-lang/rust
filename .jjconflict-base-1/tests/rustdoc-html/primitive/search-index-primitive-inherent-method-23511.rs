#![feature(rustc_attrs)]
#![feature(rustdoc_internals)]
#![no_std]

// https://github.com/rust-lang/rust/issues/23511
#![crate_name="issue_23511"]

#[rustc_doc_primitive = "str"]
const _: () = ();

impl str {
    //@ hasraw search.index/name/*.js foo
    #[rustc_allow_incoherent_impl]
    pub fn foo(&self) {}
}
