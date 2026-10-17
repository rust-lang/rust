#[inline(never)]
#[cfg(any(rpass1, rpass3))]
fn a() -> i32 {
    0
}

#[cfg(any(cfail2))]
fn a() -> i32 {
    1
}

fn main() {
    evil::evil();
    assert_eq!(a(), 0);
}

mod evil {
    #[cfg(any(rpass1, rpass3))]
    pub fn evil() {
        unsafe {
            std::arch::asm!("/*  */");
        }
    }

    #[cfg(any(cfail2))]
    pub fn evil() {
        unsafe {
            std::arch::asm!("missing");
        }
    }
}
