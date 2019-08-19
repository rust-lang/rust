// build-pass (FIXME(62277): could be check-pass?)
#![allow(unused_variables)]
// pretty-expanded FIXME #23616

trait Trait<Input> {
    type Output;

    fn method(&self, i: Input) -> bool { false }
}

fn main() {}
