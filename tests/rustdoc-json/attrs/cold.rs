//@ is "$.index[?(@.name=='cold_fn')].attrs" '[{"other": "#[attr = Cold]"}]'
#[cold]
pub fn cold_fn() {}
