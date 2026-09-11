// rustfmt-unstable: true
// rustfmt-normalize_doc_attributes: true

#[doc = "This comment
is split
on multiple lines"]
fn foo() {}

#[doc = " B1"]
#[doc = ""]
#[doc = " A1"]
fn bar() {}
