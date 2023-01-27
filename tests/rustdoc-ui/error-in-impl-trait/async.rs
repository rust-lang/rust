// edition:2018

pub async fn a() -> u32 {
    error::_in::async_fn()
    //~^ use of undeclared crate or module `error`
}
