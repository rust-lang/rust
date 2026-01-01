//@ edition:2021

#[must_use]
#[cold]
pub unsafe fn unsafe_fn_extern() -> usize { 1 }

#[must_use = "extern_fn_extern: some reason"]
#[deprecated]
pub extern "C" fn extern_fn_extern() -> usize { 1 }

pub const fn const_fn_extern() -> usize { 1 }

#[must_use]
pub async fn async_fn_extern() { }
