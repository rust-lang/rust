//@ build-pass
//@ compile-flags: -Zvirtual-function-elimination=true -Clto=true
//@ only-x86_64
//@ no-prefer-dynamic
//@ ignore-backends: gcc

// issue #123955
pub fn test0() {
    _ = Box::new(()) as Box<dyn Send>;
}

// issue #124092
const X: for<'b> fn(&'b ()) = |&()| ();
pub fn test1() {
    let _dyn_debug = Box::new(X) as Box<fn(&'static ())> as Box<dyn Send>;
}

fn main() {}
