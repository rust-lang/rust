struct Foo;

fn foo_accepting_func(f: Foo) {}

fn bool_accepting_func(b: bool) {}

fn char_accepting_func(c: char) {}

fn i32_accepting_func(i: i32) {}

fn float_accepting_func(i: f32) {}

fn u32_accepting_func(i: u32) {}

fn main() {
    let i: i64 = 5;
    foo_accepting_func(i); //~  ERROR mismatched types
                           //~| expected `Foo`
                           //~| found `i64`
                           //~| expected struct `Foo`
                           //~| found i64
    bool_accepting_func(i); //~  ERROR mismatched types
                            //~| expected `bool`
                            //~| found `i64`
    char_accepting_func(i); //~  ERROR mismatched types
                            //~| expected `char`
                            //~| found `i64`
    i32_accepting_func(i); //~  ERROR mismatched types
                           //~| expected `i32`
                           //~| found `i64`
    float_accepting_func(i); //~  ERROR mismatched types
                             //~| expected `f32`
                             //~| found `i64`
    u32_accepting_func(i); //~  ERROR mismatched types
                           //~| expected `u32`
                           //~| found `i64`
}
