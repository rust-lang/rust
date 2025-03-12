//@ edition:2021

pub trait BleRadio<'a> {
    async fn transmit(&mut self);
}
