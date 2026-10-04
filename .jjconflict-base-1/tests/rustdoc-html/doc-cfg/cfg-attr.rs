// This test ensures that the `cfg_attr` cfg predicates are correctly kept to be used
// by the `doc_cfg` feature.

#![crate_name = "foo"]
#![feature(doc_cfg)]

//@ has 'foo/struct.Test.html'
//@ has - '//*[@id="impl-Debug-for-Test"]/*[@class="item-info"]/*[@class="stab portability"]' \
//  'Available on non-crate feature debug only.'
//@ has - '//*[@id="impl-Clone-for-Test"]/*[@class="item-info"]/*[@class="stab portability"]' \
//  'Available on non-crate feature aa and non-crate feature bb only.'
#[cfg_attr(not(feature = "debug"), derive(Debug))]
#[cfg_attr(not(feature = "aa"), cfg_attr(not(feature = "bb"), derive(Clone)))]
pub struct Test;
