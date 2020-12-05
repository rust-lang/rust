fn main() {
    let foo = 42u32;
    const FOO : u32 = foo;
                   //~^ ERROR attempt to use a non-constant value in a constant
}
