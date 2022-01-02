// build-pass
// compile-flags:-Zpolymorphize=on -Csymbol-mangling-version=v0

fn foo(f: impl Fn()) {
    // Mutate an upvar from `x` so that it implements `FnMut`.
    let mut outer = 3;
    let mut x = |_: ()| {
        outer = 4;
        ()
    };

    // Don't use `f` in `y`, but refer to `x` so that the closure substs contain a reference to
    // `x` that will differ for each instantiation despite polymorphisation of the varying
    // argument.
    let mut y = || x(());

    // Consider `f` used in `foo`.
    f();
    // Use `y` so that it is visited in monomorphisation collection.
    y();
}

fn entry_a() {
    foo(|| ());
}

fn entry_b() {
    foo(|| ());
}

fn main() {
    entry_a();
    entry_b();
}
