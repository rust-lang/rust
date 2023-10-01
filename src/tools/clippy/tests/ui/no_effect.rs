#![feature(fn_traits, unboxed_closures)]
#![warn(clippy::no_effect_underscore_binding)]
#![allow(dead_code, path_statements)]
#![allow(
    clippy::deref_addrof,
    clippy::redundant_field_names,
    clippy::uninlined_format_args,
    clippy::unnecessary_struct_initialization,
    clippy::useless_vec
)]

struct Unit;
struct Tuple(i32);
struct Struct {
    field: i32,
}
enum Enum {
    Tuple(i32),
    Struct { field: i32 },
}
struct DropUnit;
impl Drop for DropUnit {
    fn drop(&mut self) {}
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
union Union {
    a: u8,
    b: f64,
}

fn get_number() -> i32 {
    0
}
fn get_struct() -> Struct {
    Struct { field: 0 }
}
fn get_drop_struct() -> DropStruct {
    DropStruct { field: 0 }
}

unsafe fn unsafe_fn() -> i32 {
    0
}

struct GreetStruct1;

impl FnOnce<(&str,)> for GreetStruct1 {
    type Output = ();

    extern "rust-call" fn call_once(self, (who,): (&str,)) -> Self::Output {
        println!("hello {}", who);
    }
}

struct GreetStruct2();

impl FnOnce<(&str,)> for GreetStruct2 {
    type Output = ();

    extern "rust-call" fn call_once(self, (who,): (&str,)) -> Self::Output {
        println!("hello {}", who);
    }
}

struct GreetStruct3;

impl FnOnce<(&str,)> for GreetStruct3 {
    type Output = ();

    extern "rust-call" fn call_once(self, (who,): (&str,)) -> Self::Output {
        println!("hello {}", who);
    }
}

fn main() {
    let s = get_struct();
    let s2 = get_struct();

    0;
    //~^ ERROR: statement with no effect
    //~| NOTE: `-D clippy::no-effect` implied by `-D warnings`
    s2;
    //~^ ERROR: statement with no effect
    Unit;
    //~^ ERROR: statement with no effect
    Tuple(0);
    //~^ ERROR: statement with no effect
    Struct { field: 0 };
    //~^ ERROR: statement with no effect
    Struct { ..s };
    //~^ ERROR: statement with no effect
    Union { a: 0 };
    //~^ ERROR: statement with no effect
    Enum::Tuple(0);
    //~^ ERROR: statement with no effect
    Enum::Struct { field: 0 };
    //~^ ERROR: statement with no effect
    5 + 6;
    //~^ ERROR: statement with no effect
    *&42;
    //~^ ERROR: statement with no effect
    &6;
    //~^ ERROR: statement with no effect
    (5, 6, 7);
    //~^ ERROR: statement with no effect
    ..;
    //~^ ERROR: statement with no effect
    5..;
    //~^ ERROR: statement with no effect
    ..5;
    //~^ ERROR: statement with no effect
    5..6;
    //~^ ERROR: statement with no effect
    5..=6;
    //~^ ERROR: statement with no effect
    [42, 55];
    //~^ ERROR: statement with no effect
    [42, 55][1];
    //~^ ERROR: statement with no effect
    (42, 55).1;
    //~^ ERROR: statement with no effect
    [42; 55];
    //~^ ERROR: statement with no effect
    [42; 55][13];
    //~^ ERROR: statement with no effect
    let mut x = 0;
    || x += 5;
    //~^ ERROR: statement with no effect
    let s: String = "foo".into();
    FooString { s: s };
    //~^ ERROR: statement with no effect
    let _unused = 1;
    //~^ ERROR: binding to `_` prefixed variable with no side-effect
    //~| NOTE: `-D clippy::no-effect-underscore-binding` implied by `-D warnings`
    let _penguin = || println!("Some helpful closure");
    //~^ ERROR: binding to `_` prefixed variable with no side-effect
    let _duck = Struct { field: 0 };
    //~^ ERROR: binding to `_` prefixed variable with no side-effect
    let _cat = [2, 4, 6, 8][2];
    //~^ ERROR: binding to `_` prefixed variable with no side-effect

    #[allow(clippy::no_effect)]
    0;

    // Do not warn
    get_number();
    unsafe { unsafe_fn() };
    let _used = get_struct();
    let _x = vec![1];
    DropUnit;
    DropStruct { field: 0 };
    DropTuple(0);
    DropEnum::Tuple(0);
    DropEnum::Struct { field: 0 };
    GreetStruct1("world");
    GreetStruct2()("world");
    GreetStruct3 {}("world");
}
