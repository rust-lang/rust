mod foo {
    use spam::*; //~ ERROR unresolved import `spam` [E0432]
}

fn main() {
    // Expect this to pass because the compiler knows there's a failed `*` import in `foo` that
    // might have caused it.
    foo::bar();
    // FIXME: these two should *fail* because they can't be fixed by fixing the glob import in `foo`
    ham(); // should error but doesn't
    eggs(); // should error but doesn't
}
