//@ only-x86_64

//@ is "$.index[?(@.name=='test1')].attrs" '["#[target_feature(enable=\"avx\")]"]'
//@ is "$.index[?(@.name=='test1')].inner.function.header.is_unsafe" false
#[target_feature(enable = "avx")]
pub fn test1() {}

//@ is "$.index[?(@.name=='test2')].attrs" '["#[target_feature(enable=\"avx\", enable=\"avx2\")]"]'
//@ is "$.index[?(@.name=='test1')].inner.function.header.is_unsafe" false
#[target_feature(enable = "avx,avx2")]
pub fn test2() {}

//@ is "$.index[?(@.name=='test3')].attrs" '["#[target_feature(enable=\"avx\", enable=\"avx2\")]"]'
//@ is "$.index[?(@.name=='test1')].inner.function.header.is_unsafe" false
#[target_feature(enable = "avx", enable = "avx2")]
pub fn test3() {}

//@ is "$.index[?(@.name=='test4')].attrs" '["#[target_feature(enable=\"avx\", enable=\"avx2\", enable=\"avx512f\")]"]'
//@ is "$.index[?(@.name=='test1')].inner.function.header.is_unsafe" false
#[target_feature(enable = "avx", enable = "avx2,avx512f")]
pub fn test4() {}

//@ is "$.index[?(@.name=='test_unsafe_fn')].attrs" '["#[target_feature(enable=\"avx\")]"]'
//@ is "$.index[?(@.name=='test_unsafe_fn')].inner.function.header.is_unsafe" true
#[target_feature(enable = "avx")]
pub unsafe fn test_unsafe_fn() {}

pub struct Example;

impl Example {
    //@ is "$.index[?(@.name=='safe_assoc_fn')].attrs" '["#[target_feature(enable=\"avx\")]"]'
    //@ is "$.index[?(@.name=='safe_assoc_fn')].inner.function.header.is_unsafe" false
    #[target_feature(enable = "avx")]
    pub fn safe_assoc_fn() {}

    //@ is "$.index[?(@.name=='unsafe_assoc_fn')].attrs" '["#[target_feature(enable=\"avx\")]"]'
    //@ is "$.index[?(@.name=='unsafe_assoc_fn')].inner.function.header.is_unsafe" true
    #[target_feature(enable = "avx")]
    pub unsafe fn unsafe_assoc_fn() {}
}
