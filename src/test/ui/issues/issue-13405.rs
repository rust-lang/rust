// build-pass (FIXME(62277): could be check-pass?)
#![allow(dead_code)]
#![allow(unused_variables)]
// pretty-expanded FIXME #23616

struct Foo<'a> {
    i: &'a bool,
    j: Option<&'a isize>,
}

impl<'a> Foo<'a> {
    fn bar(&mut self, j: &isize) {
        let child = Foo {
            i: self.i,
            j: Some(j)
        };
    }
}

fn main() {}
