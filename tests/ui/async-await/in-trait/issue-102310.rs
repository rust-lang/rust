//@ check-pass
//@ edition:2021

pub trait SpiDevice {
    async fn transaction<F, R>(&mut self);
}

impl SpiDevice for () {
    async fn transaction<F, R>(&mut self) {}
}

fn main() {}
