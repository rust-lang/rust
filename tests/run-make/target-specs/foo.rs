#![feature(lang_items, no_core, auto_traits)]
#![no_core]

#[lang = "copy"]
trait Copy {}

#[lang = "pointee_sized"]
pub trait PointeeSized {}

#[lang = "size_of_val"]
pub trait SizeOfVal: PointeeSized {}

#[lang = "sized"]
trait Sized: SizeOfVal {}

#[lang = "freeze"]
auto trait Freeze {}

#[lang = "start"]
fn start<T>(_main: fn() -> T, _argc: isize, _argv: *const *const u8, _sigpipe: u8) -> isize {
    0
}

extern "C" {
    fn _foo() -> [u8; 16];
}

fn _main() {
    let _a = unsafe { _foo() };
}
