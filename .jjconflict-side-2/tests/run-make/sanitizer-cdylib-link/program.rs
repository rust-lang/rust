#[cfg_attr(windows, link(name = "library.dll.lib", modifiers = "+verbatim"))]
#[cfg_attr(not(windows), link(name = "library"))]
extern "C" {
    fn overflow();
}

fn main() {
    unsafe { overflow() }
}
