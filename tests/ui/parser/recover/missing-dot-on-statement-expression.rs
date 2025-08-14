//@ run-rustfix
#![allow(unused_must_use, dead_code)]
struct S {
    field: (),
}
fn main() {
    let _ = [1, 2, 3].iter()map(|x| x); //~ ERROR expected one of `.`, `;`, `?`, `else`, or an operator, found `map`
    //~^ HELP you might have meant to write a method call
}
fn foo() {
    let baz = S {
        field: ()
    };
    let _ = baz field; //~ ERROR expected one of `!`, `.`, `::`, `;`, `?`, `else`, `{`, or an operator, found `field`
    //~^ HELP you might have meant to write a field
}

fn bar() {
    [1, 2, 3].iter()map(|x| x); //~ ERROR expected one of `.`, `;`, `?`, `}`, or an operator, found `map`
    //~^ HELP you might have meant to write a method call
}
fn baz() {
    let baz = S {
        field: ()
    };
    baz field; //~ ERROR expected one of `!`, `.`, `::`, `;`, `?`, `{`, `}`, or an operator, found `field`
    //~^ HELP you might have meant to write a field
}
