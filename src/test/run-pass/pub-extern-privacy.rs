use std::cast::transmute;

mod a {
    extern {
        pub fn free(x: *u8);
    }
}

fn main() {
    unsafe {
        a::free(transmute(0));
    }
}

