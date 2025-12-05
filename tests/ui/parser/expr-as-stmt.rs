//@ run-rustfix
//@ rustfix-only-machine-applicable
#![allow(unused_variables)]
#![allow(dead_code)]
#![allow(unused_must_use)]

fn foo() -> i32 {
    {2} + {2} //~ ERROR expected expression, found `+`
    //~^ ERROR mismatched types
}

fn bar() -> i32 {
    {2} + 2 //~ ERROR leading `+` is not supported
    //~^ ERROR mismatched types
}

fn zul() -> u32 {
    let foo = 3;
    { 42 } + foo; //~ ERROR expected expression, found `+`
    //~^ ERROR mismatched types
    32
}

fn baz() -> i32 {
    { 3 } * 3 //~ ERROR type `{integer}` cannot be dereferenced
    //~^ ERROR mismatched types
}

fn moo(x: u32) -> bool {
    match x {
        _ => 1,
    } > 0 //~ ERROR expected expression
}

fn qux() -> u32 {
    {2} - 2 //~ ERROR cannot apply unary operator `-` to type `u32`
    //~^ ERROR mismatched types
}

fn space_cadet() -> bool {
    { true } | { true } //~ ERROR E0308
    //~^ ERROR expected parameter name
}

fn revenge_from_mars() -> bool {
    { true } && { true } //~ ERROR E0308
    //~^ ERROR mismatched types
}

fn attack_from_mars() -> bool {
    { true } || { true } //~ ERROR E0308
    //~^ ERROR mismatched types
}

// This gets corrected by adding a semicolon, instead of parens.
// It's placed here to help keep track of the way this diagnostic
// needs to interact with type checking to avoid MachineApplicable
// suggestions that actually break stuff.
//
// If you're wondering what happens if that `foo()` is a `true` like
// all the ones above use? Nothing. It makes neither suggestion in
// that case.
fn asteroids() -> impl FnOnce() -> bool {
    { foo() } || { true } //~ ERROR E0308
}

// https://github.com/rust-lang/rust/issues/105179
fn r#match() -> i32 {
    match () { () => 1 } + match () { () => 1 } //~ ERROR expected expression, found `+`
    //~^ ERROR mismatched types
}

// https://github.com/rust-lang/rust/issues/102171
fn r#unsafe() -> i32 {
    unsafe { 1 } + unsafe { 1 } //~ ERROR expected expression, found `+`
    //~^ ERROR mismatched types
}

// https://github.com/rust-lang/rust/issues/88727
fn matches() -> bool {
    match () { _ => true } && match () { _ => true }; //~ ERROR mismatched types
    match () { _ => true } && match () { _ => true } //~ ERROR mismatched types
    //~^ ERROR expected `;`, found keyword `match`
    match () { _ => true } && true; //~ ERROR mismatched types
    match () { _ => true } && true //~ ERROR mismatched types
    //~^ ERROR mismatched types
}
fn main() {}
