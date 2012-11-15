// xfail-test
mod a {
fn foo(f: fn&()) { f() }
fn bar() {}
fn main() { foo(||bar()); }
}

mod b {
fn foo(f: Option<fn&()>) { f.iter(|x|x()) }
fn bar() {}
fn main() { foo(Some(bar)); }
}

mod c {
fn foo(f: Option<fn&()>) { f.iter(|x|x()) }
fn bar() {}
fn main() { foo(Some(||bar())); }
}

fn main() {
}