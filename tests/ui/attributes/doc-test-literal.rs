#![deny(invalid_doc_attributes)]

#![doc(test(""))]
//~^ ERROR
//~| WARN

fn main() {}
