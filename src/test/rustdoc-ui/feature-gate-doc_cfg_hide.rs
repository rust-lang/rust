#![doc(cfg_hide(test))]
//~^ ERROR `#[doc(cfg_hide)]` is experimental

#[cfg(not(test))]
pub fn public_fn() {}
#[cfg(test)]
pub fn internal_use_only() {}
