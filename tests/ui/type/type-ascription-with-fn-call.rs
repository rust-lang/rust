//@ run-rustfix
fn main() {
    f()  : //~ ERROR statements are terminated with a semicolon
    f();
}

fn f() {}
