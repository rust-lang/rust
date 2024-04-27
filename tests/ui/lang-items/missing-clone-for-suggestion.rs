// Avoid panicking if the Clone trait is not found while building error suggestions
// See #104870

#![feature(no_core, lang_items)]
#![no_core]

#[lang = "sized"]
trait Sized {}

#[lang = "copy"]
trait Copy {}

fn g<T>(x: T) {}

fn f(x: *mut u8) {
    g(x);
    g(x); //~ ERROR use of moved value: `x`
}

fn main() {}
