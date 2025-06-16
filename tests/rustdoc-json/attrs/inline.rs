//@ is "$.index[?(@.name=='just_inline')].attrs" '["#[attr = Inline(Hint)]"]'
#[inline]
pub fn just_inline() {}

//@ is "$.index[?(@.name=='inline_always')].attrs" '["#[attr = Inline(Always)]"]'
#[inline(always)]
pub fn inline_always() {}

//@ is "$.index[?(@.name=='inline_never')].attrs" '["#[attr = Inline(Never)]"]'
#[inline(never)]
pub fn inline_never() {}
