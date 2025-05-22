//@ is "$.index[?(@.name=='just_inline')].attrs" '["#[inline]"]'
#[inline]
pub fn just_inline() {}

//@ is "$.index[?(@.name=='inline_always')].attrs" '["#[inline(always)]"]'
#[inline(always)]
pub fn inline_always() {}

//@ is "$.index[?(@.name=='inline_never')].attrs" '["#[inline(never)]"]'
#[inline(never)]
pub fn inline_never() {}
