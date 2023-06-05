#![feature(builtin_syntax)]

// For the exposed macro we already test these errors in the other files,
// but this test helps to make sure the builtin construct also errors.
// This has the same examples as offset-of-arg-count.rs

fn main() {
    builtin # offset_of(NotEnoughArguments); //~ ERROR expected one of
}
fn t1() {
    // Already errored upon at the macro level. Yielding an error would require
    // extra effort.
    builtin # offset_of(NotEnoughArgumentsWithAComma, );
}
fn t2() {
    builtin # offset_of(Container, field, too many arguments); //~ ERROR expected identifier, found
    //~| ERROR found `,`
    //~| ERROR found `many`
    //~| ERROR found `arguments`
}
fn t3() {
    builtin # offset_of(S, f); // compiles fine
}
fn t4() {
    // Already errored upon at the macro level. Yielding an error would require
    // extra effort.
    builtin # offset_of(S, f);
}
fn t5() {
    builtin # offset_of(S, f.); //~ ERROR expected identifier
}
fn t6() {
    builtin # offset_of(S, f.,); //~ ERROR expected identifier
}
fn t7() {
    builtin # offset_of(S, f..); //~ ERROR expected one of
}
fn t8() {
    // Already errored upon at the macro level. Yielding an error would require
    // extra effort.
    builtin # offset_of(S, f..,);
}

struct S { f: u8, }
