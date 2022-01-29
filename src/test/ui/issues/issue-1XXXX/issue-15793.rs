// run-pass
#![allow(dead_code)]

enum NestedEnum {
    First,
    Second,
    Third
}
enum Enum {
    Variant1(bool),
    Variant2(NestedEnum)
}

#[inline(never)]
fn foo(x: Enum) -> isize {
    match x {
        Enum::Variant1(true) => 1,
        Enum::Variant1(false) => 2,
        Enum::Variant2(NestedEnum::Second) => 3,
        Enum::Variant2(NestedEnum::Third) => 4,
        Enum::Variant2(NestedEnum::First) => 5
    }
}

fn main() {
    assert_eq!(foo(Enum::Variant2(NestedEnum::Third)), 4);
}
