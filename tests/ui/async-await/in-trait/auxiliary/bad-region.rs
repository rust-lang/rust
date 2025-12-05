//@ edition:2021

#[allow(async_fn_in_trait)]

pub trait BleRadio<'a> {
    async fn transmit(&mut self);
}
