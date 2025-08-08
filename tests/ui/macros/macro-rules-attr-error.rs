#![feature(macro_attr)]

macro_rules! local_attr {
    attr() { $($body:tt)* } => {
        compile_error!(concat!("local_attr: ", stringify!($($body)*)));
    };
    //~^^ ERROR: local_attr
}

fn main() {
    #[local_attr]
    struct S;

    local_attr!(arg); //~ ERROR: macro has no rules for function-like invocation
}
