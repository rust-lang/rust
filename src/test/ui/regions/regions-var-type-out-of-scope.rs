fn id<T>(x: T) -> T { x }

fn foo(cond: bool) {
    // Here we will infer a type that uses the
    // region of the if stmt then block:
    let mut x;

    if cond {
        x = &id(3); //~ ERROR borrowed value does not live long enough
        assert_eq!(*x, 3);
    }
}

fn main() {}
