//@ normalize-stderr: "\d+ bits" -> "N bits"

trait Bar {
    type Bar;
}

struct ArrayPeano<T: Bar> {
    data: T::Bar,
}

fn foo<T>(a: &ArrayPeano<T>) -> &[T] where T: Bar {
    unsafe { std::mem::transmute(a) } //~ ERROR cannot transmute between types of different sizes
}

impl Bar for () {
    type Bar = ();
}

fn main() {
    let x: ArrayPeano<()> = ArrayPeano { data: () };
    foo(&x);
}
