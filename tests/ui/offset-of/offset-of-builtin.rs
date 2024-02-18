#![feature(builtin_syntax)]

// For the exposed macro we already test these errors in the other files,
// but this test helps to make sure the builtin construct also errors.
// This has the same examples as offset-of-arg-count.rs

fn main() {
    builtin # offset_of(NotEnoughArguments); //~ ERROR expected one of
}
fn t1() {
    builtin # offset_of(NotEnoughArgumentsWithAComma, ); //~ ERROR expected expression
}
fn t2() {
    builtin # offset_of(S, f, too many arguments); //~ ERROR expected `)`, found `too`
}
fn t3() {
    builtin # offset_of(S, f); // compiles fine
}
fn t4() {
    builtin # offset_of(S, f.); //~ ERROR unexpected token
}
fn t5() {
    builtin # offset_of(S, f.,); //~ ERROR unexpected token
}
fn t6() {
    builtin # offset_of(S, f..); //~ ERROR offset_of expects dot-separated field and variant names
}
fn t7() {
    builtin # offset_of(S, f..,); //~ ERROR offset_of expects dot-separated field and variant names
}

struct S { f: u8, }
