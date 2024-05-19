enum MyEnum {
    Tuple(i32),
    Struct{ s: i32 },
}

fn foo(en: MyEnum) {
    match en {
        MyEnum::Tuple => "",
//~^ ERROR expected unit struct, unit variant or constant, found tuple variant `MyEnum::Tuple`
        MyEnum::Struct => "",
//~^ ERROR expected unit struct, unit variant or constant, found struct variant `MyEnum::Struct`
    };
}

fn main() {}
