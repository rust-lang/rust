#![deny(non_snake_case)]
#![allow(dead_code)]

struct Foo;

impl Foo {
    fn Foo_Method() {}
    //~^ ERROR method `Foo_Method` should have a snake case name

    // Don't allow two underscores in a row
    fn foo__method(&self) {}
    //~^ ERROR method `foo__method` should have a snake case name

    pub fn xyZ(&mut self) {}
    //~^ ERROR method `xyZ` should have a snake case name

    fn render_HTML() {}
    //~^ ERROR method `render_HTML` should have a snake case name
}

trait X {
    fn ABC();
    //~^ ERROR trait method `ABC` should have a snake case name

    fn a_b_C(&self) {}
    //~^ ERROR trait method `a_b_C` should have a snake case name

    fn something__else(&mut self);
    //~^ ERROR trait method `something__else` should have a snake case name
}

impl X for Foo {
    // These errors should be caught at the trait definition not the impl
    fn ABC() {}
    fn something__else(&mut self) {}
}

fn Cookie() {}
//~^ ERROR function `Cookie` should have a snake case name

pub fn bi_S_Cuit() {}
//~^ ERROR function `bi_S_Cuit` should have a snake case name

fn main() { }
