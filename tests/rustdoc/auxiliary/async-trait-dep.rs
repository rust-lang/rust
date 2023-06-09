// edition:2021

#![feature(async_fn_in_trait)]
#![allow(incomplete_features)]

pub trait Meow {
    /// Who's a good dog?
    async fn woof();
}
