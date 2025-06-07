//@ normalize-stderr: "you are using [0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9]+)?( \([^)]*\))?" -> "you are using $$RUSTC_VERSION"

#![feature(external_doc)] //~ ERROR feature has been removed
#![doc(include("README.md"))] //~ ERROR unknown `doc` attribute `include`

fn main(){}
