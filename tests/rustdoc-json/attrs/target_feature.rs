//@ only-x86_64

//@ eq .index[] | select(.name == "test1") | [.attrs, .inner.function.header?.is_unsafe], [["#[target_feature(enable=\"avx\")]"], false]
#[target_feature(enable = "avx")]
pub fn test1() {}

//@ eq .index[] | select(.name == "test2") | [.attrs, .inner.function.header?.is_unsafe], [["#[target_feature(enable=\"avx\", enable=\"avx2\")]"], false]
#[target_feature(enable = "avx,avx2")]
pub fn test2() {}

//@ eq .index[] | select(.name == "test3") | [.attrs, .inner.function.header?.is_unsafe], [["#[target_feature(enable=\"avx\", enable=\"avx2\")]"], false]
#[target_feature(enable = "avx", enable = "avx2")]
pub fn test3() {}

//@ eq .index[] | select(.name == "test4") | [.attrs, .inner.function.header?.is_unsafe], [["#[target_feature(enable=\"avx\", enable=\"avx2\", enable=\"avx512f\")]"], false]
#[target_feature(enable = "avx", enable = "avx2,avx512f")]
pub fn test4() {}

//@ eq .index[] | select(.name == "test_unsafe_fn") | [.attrs, .inner.function.header?.is_unsafe], [["#[target_feature(enable=\"avx\")]"], true]
#[target_feature(enable = "avx")]
pub unsafe fn test_unsafe_fn() {}

pub struct Example;

impl Example {
    //@ eq .index[] | select(.name == "safe_assoc_fn") | [.attrs, .inner.function.header?.is_unsafe], [["#[target_feature(enable=\"avx\")]"], false]
    #[target_feature(enable = "avx")]
    pub fn safe_assoc_fn() {}

    //@ eq .index[] | select(.name == "unsafe_assoc_fn") | [.attrs, .inner.function.header?.is_unsafe], [["#[target_feature(enable=\"avx\")]"], true]
    #[target_feature(enable = "avx")]
    pub unsafe fn unsafe_assoc_fn() {}
}
