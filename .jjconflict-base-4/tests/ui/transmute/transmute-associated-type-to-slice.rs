//! regression test for <https://github.com/rust-lang/rust/issues/28625>
//@ normalize-stderr: "\d+ bits" -> "N bits"

trait MyTrait {
    type MyType;
}

struct ArrayPeano<T: MyTrait> {
    data: T::MyType,
}

fn foo<T>(a: &ArrayPeano<T>) -> &[T] where T: MyTrait {
    unsafe { std::mem::transmute(a) } //~ ERROR cannot transmute between types of different sizes
}

impl MyTrait for () {
    type MyType = ();
}

fn main() {
    let x: ArrayPeano<()> = ArrayPeano { data: () };
    foo(&x);
}
