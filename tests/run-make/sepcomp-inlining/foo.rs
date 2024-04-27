#![feature(start)]

#[inline]
fn inlined() -> u32 {
    1234
}

fn normal() -> u32 {
    2345
}

mod a {
    pub fn f() -> u32 {
        ::inlined() + ::normal()
    }
}

mod b {
    pub fn f() -> u32 {
        ::inlined() + ::normal()
    }
}

#[start]
fn start(_: isize, _: *const *const u8) -> isize {
    a::f();
    b::f();

    0
}
