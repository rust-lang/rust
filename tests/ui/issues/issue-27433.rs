//@ run-rustfix
fn main() {
    let foo = 42u32;
    #[allow(unused_variables, non_snake_case)]
    const FOO : u32 = foo;
    //~^ ERROR attempt to use a non-constant value in a constant
}
