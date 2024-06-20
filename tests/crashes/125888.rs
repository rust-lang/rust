//@ known-bug: rust-lang/rust#125888
enum NestedEnum {
    First,
    Second,
}

enum Enum {
    Variant(*const &'a ()),
}

fn foo(x: Enum) {
    match x {
        Enum::Variant(NestedEnum::Second) => {}
    }
}

fn main() {}
