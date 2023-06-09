// run-pass
#![allow(dead_code)]
#![allow(non_camel_case_types)]
// pretty-expanded FIXME #23616

struct thing { x: isize, }

impl Drop for thing {
    fn drop(&mut self) {}
}

fn thing() -> thing {
    thing {
        x: 0
    }
}

impl thing {
    pub fn f(self) {}
}

pub fn main() {
    let z = thing();
    (z).f();
}
