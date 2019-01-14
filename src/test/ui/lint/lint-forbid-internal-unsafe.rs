#![forbid(unsafe_code)]
#![feature(allow_internal_unsafe)]

#[allow_internal_unsafe]
//~^ ERROR: cannot use `allow_internal_unsafe` with `forbid(unsafe_code)`
macro_rules! evil {
    ($e:expr) => {
        unsafe {
            $e
        }
    }
}

fn main() {
    println!("{}", evil!(*(0 as *const u8)));
}
