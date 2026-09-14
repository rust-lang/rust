//@ jq_is '.index[] | select(.name == "cold_fn").attrs.[0].other' '"#[attr = Cold]"'
#[cold]
pub fn cold_fn() {}
