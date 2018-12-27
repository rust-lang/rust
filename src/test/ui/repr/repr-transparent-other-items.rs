// See also repr-transparent.rs

#[repr(transparent)] //~ ERROR unsupported representation for zero-variant enum
enum Void {}         //~| ERROR should be applied to struct

#[repr(transparent)] //~ ERROR should be applied to struct
enum FieldlessEnum {
    Foo,
    Bar,
}

#[repr(transparent)] //~ ERROR should be applied to struct
enum Enum {
    Foo(String),
    Bar(u32),
}

#[repr(transparent)] //~ ERROR should be applied to struct
union Foo {
    u: u32,
    s: i32
}

#[repr(transparent)] //~ ERROR should be applied to struct
fn cant_repr_this() {}

#[repr(transparent)] //~ ERROR should be applied to struct
static CANT_REPR_THIS: u32 = 0;

fn main() {}
