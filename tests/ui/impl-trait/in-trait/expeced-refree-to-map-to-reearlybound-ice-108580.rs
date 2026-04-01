// ICE expected ReFree to map to ReEarlyBound
// issue: rust-lang/rust#108580
//@ check-pass

trait Foo {
    fn bar(&self) -> impl Iterator<Item = impl Sized> + '_;
}

impl Foo for () {
    fn bar(&self) -> impl Iterator + '_ {
    //~^ WARN impl trait in impl method signature does not match trait method signature
        vec![()].into_iter()
    }
}

pub fn main() {}
