#![feature(doc_keyword)]

#[doc(keyword = "foo df")] //~ ERROR
mod foo {}
