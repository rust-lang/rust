struct S;

impl S {
    //~^ ERROR missing `fn`, `type`, or `const` for associated-item declaration
    static fn f() {}
}

fn main() {}
