#[cfg_attr(windows, link(name = "library", kind = "raw-dylib"))]
#[cfg_attr(not(windows), link(name = "library"))]
extern "C" {
    fn overflow();
}

fn main() {
    unsafe { overflow() }
}
