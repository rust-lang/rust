//@ only-x86_64

//@ is "$.index[?(@.name=='test1')].attrs" '["#[target_feature(enable=\"avx\")]"]'
#[target_feature(enable = "avx")]
pub fn test1() {}

//@ is "$.index[?(@.name=='test2')].attrs" '["#[target_feature(enable=\"avx\", enable=\"avx2\")]"]'
#[target_feature(enable = "avx,avx2")]
pub fn test2() {}

//@ is "$.index[?(@.name=='test3')].attrs" '["#[target_feature(enable=\"avx\", enable=\"avx2\")]"]'
#[target_feature(enable = "avx", enable = "avx2")]
pub fn test3() {}

//@ is "$.index[?(@.name=='test4')].attrs" '["#[target_feature(enable=\"avx\", enable=\"avx2\", enable=\"avx512f\")]"]'
#[target_feature(enable = "avx", enable = "avx2,avx512f")]
pub fn test4() {}
