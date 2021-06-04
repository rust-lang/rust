#![crate_type = "lib"]
#![feature(doc_keyword)]

#![doc(keyword = "hello")] //~ ERROR

#[doc(keyword = "hell")] //~ ERROR
mod foo {
    fn hell() {}
}

#[doc(keyword = "hall")] //~ ERROR
fn foo() {}
