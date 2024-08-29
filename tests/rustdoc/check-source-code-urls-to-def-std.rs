//@ compile-flags: -Zunstable-options --generate-link-to-definition

#![crate_name = "foo"]

//@ has 'src/foo/check-source-code-urls-to-def-std.rs.html'

fn babar() {}

//@ has - '//a[@href="{{channel}}/std/primitive.u32.html"]' 'u32'
//@ has - '//a[@href="{{channel}}/std/primitive.str.html"]' 'str'
//@ has - '//a[@href="{{channel}}/std/primitive.bool.html"]' 'bool'
//@ has - '//a[@href="#7"]' 'babar'
pub fn foo(a: u32, b: &str, c: String) {
    let x = 12;
    let y: bool = true;
    babar();
}

macro_rules! yolo { () => {}}

fn bar(a: i32) {}

macro_rules! bar {
    ($a:ident) => { bar($a) }
}

macro_rules! data {
    ($x:expr) => { $x * 2 }
}

pub fn another_foo() {
    // This is known limitation: if the macro doesn't generate anything, the visitor
    // can't find any item or anything that could tell us that it comes from expansion.
    //@ !has - '//a[@href="#19"]' 'yolo!'
    yolo!();
    //@ has - '//a[@href="{{channel}}/std/macro.eprintln.html"]' 'eprintln!'
    eprintln!();
    //@ has - '//a[@href="#27-29"]' 'data!'
    let x = data!(4);
    //@ has - '//a[@href="#23-25"]' 'bar!'
    bar!(x);
}
