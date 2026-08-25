// https://github.com/rust-lang/rust/issues/49153

//@ revisions:rpass1 rpass2
//@ ignore-backends: gcc

extern "C" {
    pub static __ImageBase: u8;
}

pub static FOO: &'static u8 = unsafe { &__ImageBase };

fn main() {}
