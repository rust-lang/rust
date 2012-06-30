// xfail-test

fn foo(cond: bool) {
    // Here we will infer a type that uses the
    // region of the if stmt then block:
    let mut x; //~ ERROR foo

    if cond {
        x = &3;
    }
}

fn main() {}