#![feature(doc_cfg)]

#![crate_name = "foo"]

//@ has 'foo/index.html'

//@ has   - '//dl[@class="item-table"]/dt[1]//a' 'f1_safe'
//@ has   - '//dl[@class="item-table"]/dt[1]//code' 'popcnt'
//@ count - '//dl[@class="item-table"]/dt[1]//sup' 0
//@ has   - '//dl[@class="item-table"]/dt[2]//a' 'f2_not_safe'
//@ has   - '//dl[@class="item-table"]/dt[2]//code' 'avx2'
//@ count - '//dl[@class="item-table"]/dt[2]//sup' 1
//@ has   - '//dl[@class="item-table"]/dt[2]//sup' '⚠'

#[target_feature(enable = "popcnt")]
//@ has 'foo/fn.f1_safe.html'
//@ matches - '//pre[@class="rust item-decl"]' '^pub fn f1_safe'
//@ has - '//*[@id="main-content"]/*[@class="item-info"]/*[@class="stab portability"]' \
//        'Available with target feature popcnt only.'
pub fn f1_safe() {}

//@ has 'foo/fn.f2_not_safe.html'
//@ matches - '//pre[@class="rust item-decl"]' '^pub unsafe fn f2_not_safe()'
//@ has - '//*[@id="main-content"]/*[@class="item-info"]/*[@class="stab portability"]' \
//        'Available with target feature avx2 only.'
#[target_feature(enable = "avx2")]
pub unsafe fn f2_not_safe() {}

//@ has 'foo/fn.f3_multifeatures_in_attr.html'
//@ has - '//*[@id="main-content"]/*[@class="item-info"]/*[@class="stab portability"]' \
//        'Available on target features popcnt and avx2 only.'
#[target_feature(enable = "popcnt", enable = "avx2")]
pub fn f3_multifeatures_in_attr() {}

//@ has 'foo/fn.f4_multi_attrs.html'
//@ has - '//*[@id="main-content"]/*[@class="item-info"]/*[@class="stab portability"]' \
//        'Available on target features popcnt and avx2 only.'
#[target_feature(enable = "popcnt")]
#[target_feature(enable = "avx2")]
pub fn f4_multi_attrs() {}
