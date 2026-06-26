//@ is "$.index[?(@.name=='test1')].inner.function.header.is_unsafe" false
//@ count "$.index[?(@.name=='test1')].attrs[*]" 1
//@ is    "$.index[?(@.name=='test1')].attrs[*].target_feature.enable" '["avx"]'
#[target_feature(enable = "avx")]
pub fn test1() {}

//@ is "$.index[?(@.name=='test2')].inner.function.header.is_unsafe" false
//@ count "$.index[?(@.name=='test2')].attrs[*]" 1
//@ is    "$.index[?(@.name=='test2')].attrs[*].target_feature.enable" '["avx", "avx2"]'
#[target_feature(enable = "avx,avx2")]
pub fn test2() {}

//@ is "$.index[?(@.name=='test3')].inner.function.header.is_unsafe" false
//@ count "$.index[?(@.name=='test3')].attrs[*]" 1
//@ is    "$.index[?(@.name=='test3')].attrs[*].target_feature.enable" '["avx", "avx2"]'
#[target_feature(enable = "avx", enable = "avx2")]
pub fn test3() {}

//@ is "$.index[?(@.name=='test4')].inner.function.header.is_unsafe" false
//@ count "$.index[?(@.name=='test4')].attrs[*]" 1
//@ is    "$.index[?(@.name=='test4')].attrs[*].target_feature.enable" '["avx", "avx2", "avx512f"]'
#[target_feature(enable = "avx", enable = "avx2,avx512f")]
pub fn test4() {}

//@ count "$.index[?(@.name=='test5')].attrs[*]" 1
//@ is    "$.index[?(@.name=='test5')].attrs[*].target_feature.enable" '["avx", "avx2"]'
#[target_feature(enable = "avx")]
#[target_feature(enable = "avx2")]
pub fn test5() {}

//@ is "$.index[?(@.name=='test_unsafe_fn')].inner.function.header.is_unsafe" true
//@ count "$.index[?(@.name=='test_unsafe_fn')].attrs[*]" 1
//@ is    "$.index[?(@.name=='test_unsafe_fn')].attrs[*].target_feature.enable" '["avx"]'
#[target_feature(enable = "avx")]
pub unsafe fn test_unsafe_fn() {}

pub struct Example;

impl Example {
    //@ is "$.index[?(@.name=='safe_assoc_fn')].inner.function.header.is_unsafe" false
    //@ is "$.index[?(@.name=='safe_assoc_fn')].attrs[*].target_feature.enable" '["avx"]'
    #[target_feature(enable = "avx")]
    pub fn safe_assoc_fn() {}

    //@ is "$.index[?(@.name=='unsafe_assoc_fn')].inner.function.header.is_unsafe" true
    //@ is "$.index[?(@.name=='unsafe_assoc_fn')].attrs[*].target_feature.enable" '["avx"]'
    #[target_feature(enable = "avx")]
    pub unsafe fn unsafe_assoc_fn() {}
}
