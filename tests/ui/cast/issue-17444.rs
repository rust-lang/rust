enum Test {
    Foo = 0
}

fn main() {
    let _x = Test::Foo as *const isize;
    //~^ ERROR casting `Test` as `*const isize` is invalid
}
