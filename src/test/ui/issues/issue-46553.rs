// run-pass
#![feature(const_fn)]
#![deny(const_err)]

pub struct Data<T> {
    function: fn() -> T,
}

impl<T> Data<T> {
    pub const fn new(function: fn() -> T) -> Data<T> {
        Data {
            function: function,
        }
    }
}

pub static DATA: Data<i32> = Data::new(|| {
    413i32
});

fn main() {
    print!("{:?}", (DATA.function)());
}
