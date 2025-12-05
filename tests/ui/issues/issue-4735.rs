//@ run-pass

use std::mem::transmute;

struct NonCopyable(*const u8);

impl Drop for NonCopyable {
    fn drop(&mut self) {
        let NonCopyable(p) = *self;
        let _v = unsafe { transmute::<*const u8, Box<isize>>(p) };
    }
}

pub fn main() {
    let t = Box::new(0);
    let p = unsafe { transmute::<Box<isize>, *const u8>(t) };
    let _z = NonCopyable(p);
}
