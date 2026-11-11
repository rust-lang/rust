pub trait NonLocal {
    fn method<const N: i32>();
}

use std::future::Future;
pub trait R {
    fn read<const L: usize>(&mut self) -> impl Future<Output = [u8; L]>;
}
