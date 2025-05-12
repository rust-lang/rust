//@ edition:2018
//@ check-pass

/// Should compile fine
pub async fn a() -> u32 {
    error::_in::async_fn()
}
