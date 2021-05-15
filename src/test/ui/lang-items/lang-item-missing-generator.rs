// error-pattern: requires `generator` lang_item
#![feature(no_core, lang_items, unboxed_closures)]
#![no_core]

#[lang = "sized"] pub trait Sized { }

#[lang = "fn_once"]
#[rustc_paren_sugar]
pub trait FnOnce<Args> {
    type Output;

    extern "rust-call" fn call_once(self, args: Args) -> Self::Output;
}

pub fn abc() -> impl FnOnce(f32) {
    |_| {}
}

fn main() {}
