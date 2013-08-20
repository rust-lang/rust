use std::cast::transmute;

mod a {
    extern {
        pub fn free(x: *u8);
    }
}

#[fixed_stack_segment] #[inline(never)]
fn main() {
    unsafe {
        a::free(transmute(0));
    }
}
