#![feature(c_unwind)]

#[link(name = "bar")]
extern "C-unwind" {
    fn panic();
}

fn main() {
    let _ = std::panic::catch_unwind(|| {
        unsafe { panic() };
    });
}
