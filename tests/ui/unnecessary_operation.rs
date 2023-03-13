// run-rustfix

#![allow(clippy::deref_addrof, dead_code, unused, clippy::no_effect)]
#![warn(clippy::unnecessary_operation)]

struct Tuple(i32);
struct Struct {
    field: i32,
}
enum Enum {
    Tuple(i32),
    Struct { field: i32 },
}
struct DropStruct {
    field: i32,
}
impl Drop for DropStruct {
    fn drop(&mut self) {}
}
struct DropTuple(i32);
impl Drop for DropTuple {
    fn drop(&mut self) {}
}
enum DropEnum {
    Tuple(i32),
    Struct { field: i32 },
}
impl Drop for DropEnum {
    fn drop(&mut self) {}
}
struct FooString {
    s: String,
}

fn get_number() -> i32 {
    0
}

fn get_usize() -> usize {
    0
}
fn get_struct() -> Struct {
    Struct { field: 0 }
}
fn get_drop_struct() -> DropStruct {
    DropStruct { field: 0 }
}

fn main() {
    Tuple(get_number());
    Struct { field: get_number() };
    Struct { ..get_struct() };
    Enum::Tuple(get_number());
    Enum::Struct { field: get_number() };
    5 + get_number();
    *&get_number();
    &get_number();
    (5, 6, get_number());
    get_number()..;
    ..get_number();
    5..get_number();
    [42, get_number()];
    [42, 55][get_usize()];
    (42, get_number()).1;
    [get_number(); 55];
    [42; 55][get_usize()];
    {
        get_number()
    };
    FooString {
        s: String::from("blah"),
    };

    // Do not warn
    DropTuple(get_number());
    DropStruct { field: get_number() };
    DropStruct { field: get_number() };
    DropStruct { ..get_drop_struct() };
    DropEnum::Tuple(get_number());
    DropEnum::Struct { field: get_number() };

    // Issue #9954
    fn one() -> i8 {
        1
    }
    macro_rules! use_expr {
        ($($e:expr),*) => {{ $($e;)* }}
    }
    use_expr!(isize::MIN / -(one() as isize), i8::MIN / -one());
}
