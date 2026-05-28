//@ run-rustfix
enum Enum {
    Variant(i32),
}
struct Struct(i32);

fn foo(x: Enum) -> i32 {
    if let Enum::Variant(value) = x { //~ ERROR `if` may be missing an `else` clause
        value
    }
}
fn bar(x: Enum) -> i32 {
    if let Enum::Variant(value) = x { //~ ERROR `if` may be missing an `else` clause
        let x = value + 1;
        x
    }
}
fn baz(x: Struct) -> i32 {
    if let Struct(value) = x { //~ ERROR `if` may be missing an `else` clause
        let x = value + 1;
        x
    }
}
fn main() {
    let _ = foo(Enum::Variant(42));
    let _ = bar(Enum::Variant(42));
    let _ = baz(Struct(42));
}
