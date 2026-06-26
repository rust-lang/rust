#![deny(invalid_doc_attributes)]
#![feature(external_doc)] //~ ERROR feature has been removed
#![doc(include("README.md"))] //~ ERROR unknown `doc` attribute `include`

fn main(){}
