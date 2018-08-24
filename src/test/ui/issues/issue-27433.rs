fn main() {
    let foo = 42u32;
    const FOO : u32 = foo;
                   //~^ ERROR can't capture dynamic environment
}
