#![feature(no_core)]
#![feature(lang_items)]

#![no_core]

#[cfg(target_os = "linux")]
#[link(name = "c")]
extern {}

#[lang = "start"]
fn start<T>(_main: fn() -> T, _argc: isize, _argv: *const *const u8) -> isize {
    //~^ ERROR lang item `start` function has wrong type [E0308]
    40+2
}

#[lang = "sized"]
pub trait Sized {}
#[lang = "copy"]
pub trait Copy {}

#[lang = "drop_in_place"]
#[allow(unconditional_recursion)]
pub unsafe fn drop_in_place<T: ?Sized>(to_drop: *mut T) {
    drop_in_place(to_drop)
}

#[lang = "add"]
trait Add<RHS> {
    type Output;
    fn add(self, other: RHS) -> Self::Output;
}

impl Add<isize> for isize {
    type Output = isize;
    fn add(self, other: isize) -> isize {
        self + other
    }
}

fn main() {}
