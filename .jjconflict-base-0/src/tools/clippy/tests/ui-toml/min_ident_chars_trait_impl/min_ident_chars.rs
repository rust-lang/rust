#![warn(clippy::min_ident_chars)]

trait ShortParams {
    fn f(g: i32);
    //~^ min_ident_chars
    //~| min_ident_chars
}

struct MyStruct;

impl ShortParams for MyStruct {
    fn f(g: i32) {}
    //~^ min_ident_chars
    //~| min_ident_chars
}

impl core::fmt::Display for MyStruct {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        //~^ min_ident_chars
        write!(f, "MyStruct")
    }
}

fn main() {}
