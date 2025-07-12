//@ eq .index[] | select(.name == "cold_fn").attrs | ., ["#[attr = Cold]"]
#[cold]
pub fn cold_fn() {}
