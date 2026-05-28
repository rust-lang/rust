//@ run-pass
// Regression test for Issue #30530: alloca's created for storing
// intermediate scratch values during brace-less match arms need to be
// initialized with their drop-flag set to "dropped" (or else we end
// up running the destructors on garbage data at the end of the
// function).

pub enum Handler {
    Default,
    #[allow(dead_code)]
    Custom(*mut Box<dyn Fn()>),
}

fn main() {
    #[allow(unused_must_use)] {
        take(Handler::Default, Box::new(main));
    }
}

#[inline(never)]
pub fn take(h: Handler, f: Box<dyn Fn()>) -> Box<dyn Fn()> {
    unsafe {
        match h {
            Handler::Custom(ptr) => *Box::from_raw(ptr),
            Handler::Default => f,
        }
    }
}
