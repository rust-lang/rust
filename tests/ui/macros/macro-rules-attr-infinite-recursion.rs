#![crate_type = "lib"]
#![feature(macro_attr)]

macro_rules! attr {
    attr() { $($body:tt)* } => {
        #[attr] $($body)*
    };
    //~^^ ERROR: recursion limit reached
}

#[attr]
struct S;
