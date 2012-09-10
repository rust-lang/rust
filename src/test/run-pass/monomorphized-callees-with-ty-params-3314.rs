use std;

trait Serializer {
}

trait Serializable {
    fn serialize<S: Serializer>(s: S);
}

impl int: Serializable {
    fn serialize<S: Serializer>(_s: S) { }
}

struct F<A> { a: A }

impl<A: Copy Serializable> F<A>: Serializable {
    fn serialize<S: Serializer>(s: S) {
        self.a.serialize(s);
    }
}

impl io::Writer: Serializer {
}

fn main() {
    do io::with_str_writer |wr| {
        let foo = F { a: 1 };
        foo.serialize(wr);

        let bar = F { a: F {a: 1 } };
        bar.serialize(wr);
    };
}
