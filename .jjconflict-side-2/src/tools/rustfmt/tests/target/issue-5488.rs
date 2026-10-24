// rustfmt-use_field_init_shorthand: true

struct MyStruct(u32);
struct AnotherStruct {
    a: u32,
}

fn main() {
    // Since MyStruct is a tuple struct, it should not be shorthanded to
    // MyStruct { 0 } even if use_field_init_shorthand is enabled.
    let instance = MyStruct { 0: 0 };

    // Since AnotherStruct is not a tuple struct, the shorthand should
    // apply.
    let a = 10;
    let instance = AnotherStruct { a };
}
