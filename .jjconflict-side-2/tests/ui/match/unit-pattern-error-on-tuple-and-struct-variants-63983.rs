// https://github.com/rust-lang/rust/issues/63983
enum MyEnum {
    HasTupleField(i32),
    HasStructField{ s: i32 },
}

fn foo(en: MyEnum) {
    match en {
        MyEnum::HasTupleField => "",
//~^ ERROR expected unit struct, unit variant or constant, found tuple variant `MyEnum::HasTupleField`
        MyEnum::HasStructField => "",
//~^ ERROR expected unit struct, unit variant or constant, found struct variant `MyEnum::HasStructField`
    };
}

fn main() {}
