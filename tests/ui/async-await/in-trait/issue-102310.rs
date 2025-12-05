//@ check-pass
//@ edition:2021

pub trait SpiDevice {
    #[allow(async_fn_in_trait)]
    async fn transaction<F, R>(&mut self);
}

impl SpiDevice for () {
    async fn transaction<F, R>(&mut self) {}
}

fn main() {}
