// run-pass
// pretty-expanded FIXME #23616

trait Serializer {
}

trait Serializable {
    fn serialize<S:Serializer>(&self, s: S);
}

impl Serializable for isize {
    fn serialize<S:Serializer>(&self, _s: S) { }
}

struct F<A> { a: A }

impl<A:Serializable> Serializable for F<A> {
    fn serialize<S:Serializer>(&self, s: S) {
        self.a.serialize(s);
    }
}

impl Serializer for isize {
}

pub fn main() {
    let foo = F { a: 1 };
    foo.serialize(1);

    let bar = F { a: F {a: 1 } };
    bar.serialize(2);
}
