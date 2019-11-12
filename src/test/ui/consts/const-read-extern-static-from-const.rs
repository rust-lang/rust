#![allow(dead_code)]

extern {
    static FOO: u8;
}

const X: u8 = unsafe { FOO };
//~^ any use of this value will cause an error [const_err]

fn main() {}
