//@rustc-env: CLIPPY_PETS_PRINT=1
//@rustc-env: CLIPPY_PRINT_MIR=1

#![warn(clippy::borrow_pats)]

fn main() {}

fn simple_const() -> u32 {
    15
}

fn fn_call() -> u32 {
    simple_const()
}

fn simple_cond() -> u32 {
    if true { 1 } else { 2 }
}

fn static_slice() -> &'static [u32] {
    &[]
}

fn static_string() -> &'static str {
    "Ducks are cool"
}

fn arg_or_default(arg: &String) -> &str {
    if arg.is_empty() { "Default" } else { arg }
}
