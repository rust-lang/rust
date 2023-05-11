#![feature(doc_cfg)]
#![feature(target_feature, cfg_target_feature)]

// @has doc_cfg/struct.Portable.html
// @!has - '//*[@id="main-content"]/*[@class="item-info"]/*[@class="stab portability"]' ''
// @has - '//*[@id="method.unix_and_arm_only_function"]' 'fn unix_and_arm_only_function()'
// @has - '//*[@class="stab portability"]' 'Available on Unix and ARM only.'
// @has - '//*[@id="method.wasi_and_wasm32_only_function"]' 'fn wasi_and_wasm32_only_function()'
// @has - '//*[@class="stab portability"]' 'Available on WASI and WebAssembly only.'
pub struct Portable;

// @has doc_cfg/unix_only/index.html \
//  '//*[@id="main-content"]/*[@class="item-info"]/*[@class="stab portability"]' \
//  'Available on Unix only.'
// @matches - '//*[@class="item-name"]//*[@class="stab portability"]' '\AARM\Z'
// @count - '//*[@class="stab portability"]' 2
#[doc(cfg(unix))]
pub mod unix_only {
    // @has doc_cfg/unix_only/fn.unix_only_function.html \
    //  '//*[@id="main-content"]/*[@class="item-info"]/*[@class="stab portability"]' \
    //  'Available on Unix only.'
    // @count - '//*[@class="stab portability"]' 1
    pub fn unix_only_function() {
        content::should::be::irrelevant();
    }

    // @has doc_cfg/unix_only/trait.ArmOnly.html \
    //  '//*[@id="main-content"]/*[@class="item-info"]/*[@class="stab portability"]' \
    //  'Available on Unix and ARM only.'
    // @count - '//*[@class="stab portability"]' 1
    #[doc(cfg(target_arch = "arm"))]
    pub trait ArmOnly {
        fn unix_and_arm_only_function();
    }

    #[doc(cfg(target_arch = "arm"))]
    impl ArmOnly for super::Portable {
        fn unix_and_arm_only_function() {}
    }
}

// @has doc_cfg/wasi_only/index.html \
//  '//*[@id="main-content"]/*[@class="item-info"]/*[@class="stab portability"]' \
//  'Available on WASI only.'
// @matches - '//*[@class="item-name"]//*[@class="stab portability"]' '\AWebAssembly\Z'
// @count - '//*[@class="stab portability"]' 2
#[doc(cfg(target_os = "wasi"))]
pub mod wasi_only {
    // @has doc_cfg/wasi_only/fn.wasi_only_function.html \
    //  '//*[@id="main-content"]/*[@class="item-info"]/*[@class="stab portability"]' \
    //  'Available on WASI only.'
    // @count - '//*[@class="stab portability"]' 1
    pub fn wasi_only_function() {
        content::should::be::irrelevant();
    }

    // @has doc_cfg/wasi_only/trait.Wasm32Only.html \
    //  '//*[@id="main-content"]/*[@class="item-info"]/*[@class="stab portability"]' \
    //  'Available on WASI and WebAssembly only.'
    // @count - '//*[@class="stab portability"]' 1
    #[doc(cfg(target_arch = "wasm32"))]
    pub trait Wasm32Only {
        fn wasi_and_wasm32_only_function();
    }

    #[doc(cfg(target_arch = "wasm32"))]
    impl Wasm32Only for super::Portable {
        fn wasi_and_wasm32_only_function() {}
    }
}

// tagging a function with `#[target_feature]` creates a doc(cfg(target_feature)) node for that
// item as well

// the portability header is different on the module view versus the full view
// @has doc_cfg/index.html
// @matches - '//*[@class="item-name"]//*[@class="stab portability"]' '\Aavx\Z'

// @has doc_cfg/fn.uses_target_feature.html
// @has - '//*[@id="main-content"]/*[@class="item-info"]/*[@class="stab portability"]' \
//        'Available with target feature avx only.'
#[target_feature(enable = "avx")]
pub unsafe fn uses_target_feature() {
    content::should::be::irrelevant();
}

// @has doc_cfg/fn.uses_cfg_target_feature.html
// @has - '//*[@id="main-content"]/*[@class="item-info"]/*[@class="stab portability"]' \
//        'Available with target feature avx only.'
#[doc(cfg(target_feature = "avx"))]
pub fn uses_cfg_target_feature() {
    uses_target_feature();
}

// multiple attributes should be allowed
// @has doc_cfg/fn.multiple_attrs.html \
//  '//*[@id="main-content"]/*[@class="item-info"]/*[@class="stab portability"]' \
//  'Available on x and y and z only.'
#[doc(cfg(x))]
#[doc(cfg(y), cfg(z))]
pub fn multiple_attrs() {}
