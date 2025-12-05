//@ compile-flags:--cfg foo

#![cfg_attr(foo, unstable(feature = "unstable_test_feature", issue = "none"))]
#![cfg_attr(not(foo), stable(feature = "test_feature", since = "1.0.0"))]
#![feature(staged_api)]
