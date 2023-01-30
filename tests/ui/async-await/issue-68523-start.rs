// edition:2018

#![feature(start)]

#[start]
pub async fn start(_: isize, _: *const *const u8) -> isize {
//~^ ERROR `start` is not allowed to be `async`
    0
}
