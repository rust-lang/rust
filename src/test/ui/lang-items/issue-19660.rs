// error-pattern: requires `copy` lang_item

#![feature(lang_items, start, no_core)]
#![no_core]

#[lang = "sized"]
trait Sized { }

struct S;

#[start]
fn main(_: isize, _: *const *const u8) -> isize {
    let _ = S;
    0
}
