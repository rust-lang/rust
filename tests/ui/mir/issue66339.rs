//@ compile-flags: -Z mir-opt-level=3
//@ build-pass

// This used to ICE in const-prop

fn foo() {
    let bar = |_| { };
    bar("a");
}

fn main() {
    foo();
}
