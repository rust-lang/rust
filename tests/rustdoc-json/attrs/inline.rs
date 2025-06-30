//@ jq .index[] | select(.name == "just_inline").attrs == ["#[attr = Inline(Hint)]"]
#[inline]
pub fn just_inline() {}

//@ jq .index[] | select(.name == "inline_always").attrs == ["#[attr = Inline(Always)]"]
#[inline(always)]
pub fn inline_always() {}

//@ jq .index[] | select(.name == "inline_never").attrs == ["#[attr = Inline(Never)]"]
#[inline(never)]
pub fn inline_never() {}
