// run-pass
#![allow(dead_code)]
// Test that the compiler considers the 'a bound declared in the
// trait. Issue #20890.

// pretty-expanded FIXME #23616

trait Foo<'a> {
    type Value: 'a;

    fn get(&self) -> &'a Self::Value;
}

fn takes_foo<'a,F: Foo<'a>>(f: &'a F) {
    // This call would be illegal, because it results in &'a F::Value,
    // and the only way we know that `F::Value : 'a` is because of the
    // trait declaration.

    f.get();
}

fn main() { }
