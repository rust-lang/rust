//@ known-bug: #124092
//@ compile-flags: -Zvirtual-function-elimination=true -Clto=true
//@ only-x86_64
const X: for<'b> fn(&'b ()) = |&()| ();
fn main() {
    let dyn_debug = Box::new(X) as Box<fn(&'static ())> as Box<dyn Send>;
}
