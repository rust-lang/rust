#[cfg_attr(not(windows), link(name = "bar"))]
#[cfg_attr(windows, link(name = "bar.dll"))]
extern "C-unwind" {
    fn panic();
}

fn main() {
    let _ = std::panic::catch_unwind(|| {
        unsafe { panic() };
    });
}
