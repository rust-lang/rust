// FIXME: Remove this file once feature is removed

#![doc(cfg_hide(test))] //~ ERROR

#[cfg(not(test))]
pub fn public_fn() {}
#[cfg(test)]
pub fn internal_use_only() {}
