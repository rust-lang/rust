#![deny(unused_macro_rules)]
// To make sure we are not hitting this
#![deny(unused_macros)]

macro_rules! num {
    (one) => { 1 };
    // Most simple (and common) case
    (two) => { compile_error!("foo"); };
    // Some nested use
    (two_) => { foo(compile_error!("foo")); };
    (three) => { 3 };
    (four) => { 4 }; //~ ERROR: rule #5 of macro
}
const _NUM: u8 = num!(one) + num!(three);

// compile_error not used as a macro invocation
macro_rules! num2 {
    (one) => { 1 };
    // Only identifier present
    (two) => { fn compile_error() {} }; //~ ERROR: rule #2 of macro
    // Only identifier and bang present
    (two_) => { compile_error! }; //~ ERROR: rule #3 of macro
    (three) => { 3 };
}
const _NUM2: u8 = num2!(one) + num2!(three);

fn main() {}
