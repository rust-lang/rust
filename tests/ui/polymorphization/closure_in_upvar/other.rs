// build-pass
// compile-flags:-Zpolymorphize=on -Csymbol-mangling-version=v0

fn y_uses_f(f: impl Fn()) {
    let x = |_: ()| ();

    let y = || {
        f();
        x(());
    };

    f();
    y();
}

fn x_uses_f(f: impl Fn()) {
    let x = |_: ()| { f(); };

    let y = || x(());

    f();
    y();
}

fn entry_a() {
    x_uses_f(|| ());
    y_uses_f(|| ());
}

fn entry_b() {
    x_uses_f(|| ());
    y_uses_f(|| ());
}

fn main() {
    entry_a();
    entry_b();
}
