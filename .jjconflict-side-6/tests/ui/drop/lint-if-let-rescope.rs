//@ run-rustfix

#![deny(if_let_rescope)]
#![feature(stmt_expr_attributes)]
#![allow(irrefutable_let_patterns, unused_parens)]

fn droppy() -> Droppy {
    Droppy
}
struct Droppy;
impl Drop for Droppy {
    fn drop(&mut self) {
        println!("dropped");
    }
}
impl Droppy {
    const fn get(&self) -> Option<u8> {
        None
    }
}

fn main() {
    if let Some(_value) = droppy().get() {
        // Should not lint
    }

    if let Some(_value) = droppy().get() {
        //~^ ERROR: `if let` assigns a shorter lifetime since Edition 2024
        //~| WARN: this changes meaning in Rust 2024
        //~| HELP: a `match` with a single arm can preserve the drop order up to Edition 2021
        // do something
    } else {
        //~^ HELP: the value is now dropped here in Edition 2024
        // do something else
    }

    if let Some(_value) = droppy().get() {
        //~^ ERROR: `if let` assigns a shorter lifetime since Edition 2024
        //~| WARN: this changes meaning in Rust 2024
        //~| HELP: a `match` with a single arm can preserve the drop order up to Edition 2021
        // do something
    } else if let Some(_value) = droppy().get() {
        //~^ HELP: the value is now dropped here in Edition 2024
        // do something else
    }
    //~^ HELP: the value is now dropped here in Edition 2024

    if droppy().get().is_some() {
        // Should not lint
    } else if let Some(_value) = droppy().get() {
        //~^ ERROR: `if let` assigns a shorter lifetime since Edition 2024
        //~| WARN: this changes meaning in Rust 2024
        //~| HELP: a `match` with a single arm can preserve the drop order up to Edition 2021
    } else if droppy().get().is_none() {
        //~^ HELP: the value is now dropped here in Edition 2024
    }

    if let Some(1) = { if let Some(_value) = Droppy.get() { Some(1) } else { None } } {
        //~^ ERROR: `if let` assigns a shorter lifetime since Edition 2024
        //~| WARN: this changes meaning in Rust 2024
        //~| HELP: the value is now dropped here in Edition 2024
        //~| HELP: a `match` with a single arm can preserve the drop order up to Edition 2021
    }

    if let () = { if let Some(_value) = Droppy.get() {} } {
        // This should not lint.
        // This `if let` sits is a tail expression of a block.
        // In Edition 2024, the temporaries are dropped before exiting the surrounding block.
    }

    #[rustfmt::skip]
    if (if let Some(_value) = droppy().get() { true } else { false }) {
        //~^ ERROR: `if let` assigns a shorter lifetime since Edition 2024
        //~| WARN: this changes meaning in Rust 2024
        //~| HELP: the value is now dropped here in Edition 2024
        //~| HELP: a `match` with a single arm can preserve the drop order up to Edition 2021
        // do something
    } else if (((if let Some(_value) = droppy().get() { true } else { false }))) {
        //~^ ERROR: `if let` assigns a shorter lifetime since Edition 2024
        //~| WARN: this changes meaning in Rust 2024
        //~| HELP: the value is now dropped here in Edition 2024
        //~| HELP: a `match` with a single arm can preserve the drop order up to Edition 2021
    }

    while let Some(_value) = droppy().get() {
        // Should not lint
        break;
    }

    while (if let Some(_value) = droppy().get() { false } else { true }) {
        //~^ ERROR: `if let` assigns a shorter lifetime since Edition 2024
        //~| WARN: this changes meaning in Rust 2024
        //~| HELP: the value is now dropped here in Edition 2024
        //~| HELP: a `match` with a single arm can preserve the drop order up to Edition 2021
    }

    if let Some(_value) = Some((droppy(), ()).1) {} else {}
    //~^ ERROR: `if let` assigns a shorter lifetime since Edition 2024
    //~| WARN: this changes meaning in Rust 2024
    //~| HELP: the value is now dropped here in Edition 2024
    //~| HELP: a `match` with a single arm can preserve the drop order up to Edition 2021

    // We want to keep the `if let`s below as direct descendents of match arms,
    // so the formatting is suppressed.
    #[rustfmt::skip]
    match droppy().get() {
        _ => if let Some(_value) = droppy().get() {},
        // Should not lint
        // There is implicitly a block surrounding the `if let`.
        // Given that it is a tail expression, the temporaries are dropped duly before
        // the execution is exiting the `match`.
    }

    if let Some(_value) = droppy().get() {}
}
