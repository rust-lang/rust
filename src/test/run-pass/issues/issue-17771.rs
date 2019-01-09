// run-pass
#![allow(dead_code)]
// pretty-expanded FIXME #23616

trait Aaa { fn dummy(&self) { } }

impl<'a> Aaa for &'a mut (Aaa + 'a) {}

struct Bar<'a> {
    writer: &'a mut (Aaa + 'a),
}

fn baz(_: &mut Aaa) {
}

fn foo<'a>(mut bar: Bar<'a>) {
    baz(&mut bar.writer);
}

fn main() {
}
