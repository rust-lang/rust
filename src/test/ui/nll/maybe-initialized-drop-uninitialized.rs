// compile-flags: -Zborrowck=mir
// build-pass (FIXME(62277): could be check-pass?)

#![allow(warnings)]

struct Wrap<'p> { p: &'p mut i32 }

impl<'p> Drop for Wrap<'p> {
    fn drop(&mut self) {
        *self.p += 1;
    }
}

fn main() {
    let mut x = 0;
    let wrap = Wrap { p: &mut x };
    std::mem::drop(wrap);
    x = 1; // OK, drop is inert
}
