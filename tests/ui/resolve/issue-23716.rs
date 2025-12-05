//@ dont-require-annotations: NOTE

static foo: i32 = 0;

fn bar(foo: i32) {}
//~^ ERROR function parameters cannot shadow statics
//~| NOTE cannot be named the same as a static

mod submod {
    pub static answer: i32 = 42;
}

use self::submod::answer;

fn question(answer: i32) {}
//~^ ERROR function parameters cannot shadow statics
//~| NOTE cannot be named the same as a static
fn main() {
}
