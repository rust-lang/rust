mod foo {
    use spam::*; //~ ERROR unresolved import `spam` [E0432]

    fn x() {
        // Expect these to pass because the compiler knows there's a failed `*` import that might
        // fix it.
        eggs();
        foo::bar();
    }
}

mod bar {
    fn z() {}
    fn x() {
        // Expect these to pass because the compiler knows there's a failed `*` import that might
        // fix it.
        foo::bar();
        z();
        // FIXME: should error but doesn't because as soon as there's a single glob import error, we
        // silence all resolve errors.
        eggs();
    }
}

mod baz {
    fn x() {
        use spam::*; //~ ERROR unresolved import `spam` [E0432]
        fn qux() {}
        qux();
        // Expect this to pass because the compiler knows there's a local failed `*` import that
        // might have caused it.
        eggs();
        // Expect this to pass because the compiler knows there's a failed `*` import in `foo` that
        // might have caused it.
        foo::bar();
    }
}

fn main() {
    // FIXME: should error but doesn't because as soon as there's a single glob import error, we
    // silence all resolve errors.
    ham();
}
