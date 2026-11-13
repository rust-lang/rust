#![deny(invalid_doc_attributes)]
#![expect(unused_attributes)]
#![doc(test(no_crate_inject))]
#![doc(test(no_crate_inject))]
//~^ ERROR
//~| WARN
