struct Foo {
    t: String
}

fn cond() -> bool { true }

fn foo<F>(_: F) where F: FnOnce() {}

fn main() {
    let pth = break; //~ ERROR: `break` outside of loop
    if cond() { continue } //~ ERROR: `continue` outside of loop

    while cond() {
        if cond() { break }
        if cond() { continue }
        foo(|| {
            if cond() { break } //~ ERROR: `break` inside of a closure
            if cond() { continue } //~ ERROR: `continue` inside of a closure
        })
    }

    let rs: Foo = Foo{t: pth};

    let unconstrained = break; //~ ERROR: `break` outside of loop
}
