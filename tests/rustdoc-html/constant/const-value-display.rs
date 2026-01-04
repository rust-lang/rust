#![crate_name = "foo"]

//@ has 'foo/constant.HOUR_IN_SECONDS.html'
//@ has - '//*[@class="rust item-decl"]//code' 'pub const HOUR_IN_SECONDS: u64 = _;'
pub const HOUR_IN_SECONDS: u64 = 60 * 60;

//@ has 'foo/constant.NEGATIVE.html'
//@ has - '//*[@class="rust item-decl"]//code' 'pub const NEGATIVE: i64 = _;'
pub const NEGATIVE: i64 = -60 * 60;
