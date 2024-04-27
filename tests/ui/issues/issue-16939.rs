// Make sure we don't ICE when making an overloaded call with the
// wrong arity.

fn _foo<F: Fn()> (f: F) {
    |t| f(t); //~ ERROR E0057
}

fn main() {}
