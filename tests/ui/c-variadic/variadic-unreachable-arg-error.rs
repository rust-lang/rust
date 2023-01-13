// check-pass

#![feature(c_variadic)]

extern "C" {
    fn foo(f: isize, x: u8, ...);
}

fn main() {
    unsafe {
        // FIXME: Ideally we could give an unreachable warning
        foo(1, loop {}, 1usize);
    }
}
