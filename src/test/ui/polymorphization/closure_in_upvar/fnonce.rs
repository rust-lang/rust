// build-pass
// compile-flags:-Zpolymorphize=on -Csymbol-mangling-version=v0

fn foo(f: impl Fn()) {
    // Move a non-copy type into `x` so that it implements `FnOnce`.
    let outer = Vec::<u32>::new();
    let x = move |_: ()| {
        let inner = outer;
        ()
    };

    // Don't use `f` in `y`, but refer to `x` so that the closure substs contain a reference to
    // `x` that will differ for each instantiation despite polymorphisation of the varying
    // argument.
    let y = || x(());

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
