#![cfg_attr(foo, experimental)]
#![cfg_attr(not(foo), stable(feature = "test_feature", since = "1.0.0"))]
#![feature(staged_api)]
