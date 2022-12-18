// error-pattern: requires `generator` lang_item
#![feature(no_core, lang_items, unboxed_closures, tuple_trait)]
#![no_core]

#[lang = "sized"] pub trait Sized { }

#[lang = "tuple_trait"] pub trait Tuple { }

#[lang = "fn_once"]
#[rustc_paren_sugar]
pub trait FnOnce<Args: Tuple> {
    type Output;

    extern "rust-call" fn call_once(self, args: Args) -> Self::Output;
}

pub fn abc() -> impl FnOnce(f32) {
    |_| {}
}

fn main() {}
