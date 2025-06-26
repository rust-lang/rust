//@ is "$.index[?(@.name=='cold_fn')].attrs" '["#[attr = Cold]"]'
#[cold]
pub fn cold_fn() {}
