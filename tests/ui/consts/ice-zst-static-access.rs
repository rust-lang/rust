//@ check-pass

// This is a regression test for ICEs from
// https://github.com/rust-lang/rust/issues/71612
// and
// https://github.com/rust-lang/rust/issues/71709

#[derive(Copy, Clone)]
pub struct Glfw;

static mut GLFW: Option<Glfw> = None;
pub fn new() -> Glfw {
    unsafe {
        if let Some(glfw) = GLFW {
            return glfw;
        } else {
            todo!()
        }
    };
}

extern "C" {
    static _dispatch_queue_attr_concurrent: [u8; 0];
}

static DISPATCH_QUEUE_CONCURRENT: &'static [u8; 0] =
    unsafe { &_dispatch_queue_attr_concurrent };

fn main() {
    *DISPATCH_QUEUE_CONCURRENT;
    new();
}
