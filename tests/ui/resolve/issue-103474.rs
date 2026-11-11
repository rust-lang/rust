struct S {}
impl S {
    fn first(&self) {}

    fn second(&self) {
        first()
        //~^ ERROR cannot find function `first` in this scope
    }

    fn third(&self) {
        no_method_err()
        //~^ ERROR cannot find function `no_method_err` in this scope
    }
}

// https://github.com/rust-lang/rust/pull/103531#discussion_r1004728080
struct Foo {
    i: i32,
}

impl Foo {
    fn needs_self() {
        this.i
        //~^ ERROR cannot find value `this` in this scope
    }
}

fn main() {}
