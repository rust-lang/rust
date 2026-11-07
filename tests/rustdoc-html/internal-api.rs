#![crate_name = "internal_api"]
#![feature(staged_api)]
#![allow(internal_features)]

#![stable(feature = "test_crate", since = "1.0.0")]

//@ has internal_api/index.html '//dt/a[@href="struct.CoreIntrinsic.html"]' 'CoreIntrinsic'
//@ has internal_api/index.html '//dt/a[@href="struct.CoreIntrinsic.html"]/../span[@class="stab internal"]' 'Internal'
//@ count internal_api/index.html '//dt/a[@href="struct.CoreIntrinsic.html"]/../span[@class="stab unstable"]' 0

//@ has internal_api/struct.CoreIntrinsic.html '//div[@class="stab internal"]//code' 'core_intrinsics'
//@ count internal_api/struct.CoreIntrinsic.html '//div[@class="stab internal"]' 1
//@ count internal_api/struct.CoreIntrinsic.html '//div[@class="stab unstable"]' 0
#[unstable(feature = "core_intrinsics", issue = "none")]
pub struct CoreIntrinsic;

//@ has internal_api/index.html '//dt/a[@href="struct.InternalIntrinsic.html"]' 'InternalIntrinsic'
//@ has internal_api/index.html '//dt/a[@href="struct.InternalIntrinsic.html"]/../span[@class="stab internal"]' 'Internal'
//@ count internal_api/index.html '//dt/a[@href="struct.InternalIntrinsic.html"]/../span[@class="stab unstable"]' 0

//@ has internal_api/struct.InternalIntrinsic.html '//div[@class="stab internal"]//code' 'foo_internal'
//@ count internal_api/struct.InternalIntrinsic.html '//div[@class="stab internal"]' 1
//@ count internal_api/struct.InternalIntrinsic.html '//div[@class="stab unstable"]' 0
#[unstable(feature = "foo_internal", issue = "none")]
pub struct InternalIntrinsic;

//@ has internal_api/index.html '//dt/a[@href="struct.InternalIntrinsics.html"]' 'InternalIntrinsics'
//@ has internal_api/index.html '//dt/a[@href="struct.InternalIntrinsics.html"]/../span[@class="stab internal"]' 'Internal'
//@ count internal_api/index.html '//dt/a[@href="struct.InternalIntrinsics.html"]/../span[@class="stab unstable"]' 0

//@ has internal_api/struct.InternalIntrinsics.html '//div[@class="stab internal"]//code' 'foo_internals'
//@ count internal_api/struct.InternalIntrinsics.html '//div[@class="stab internal"]' 1
//@ count internal_api/struct.InternalIntrinsics.html '//div[@class="stab unstable"]' 0
#[unstable(feature = "foo_internals", issue = "none")]
pub struct InternalIntrinsics;

//@ has internal_api/index.html '//dt/a[@href="struct.ExperimentalUnstable.html"]' 'ExperimentalUnstable'
//@ count internal_api/index.html '//dt/a[@href="struct.ExperimentalUnstable.html"]/../span[@class="stab internal"]' 0
//@ has internal_api/index.html '//dt/a[@href="struct.ExperimentalUnstable.html"]/../span[@class="stab unstable"]' 'Experimental'

//@ has internal_api/struct.ExperimentalUnstable.html '//div[@class="stab unstable"]//code' 'ordinary_feature'
//@ count internal_api/struct.ExperimentalUnstable.html '//div[@class="stab internal"]' 0
//@ count internal_api/struct.ExperimentalUnstable.html '//div[@class="stab unstable"]' 1
#[unstable(feature = "ordinary_feature", issue = "none")]
pub struct ExperimentalUnstable;

//@ has internal_api/index.html '//dt/a[@href="struct.ExperimentalExtra.html"]' 'ExperimentalExtra'
//@ count internal_api/index.html '//dt/a[@href="struct.ExperimentalExtra.html"]/../span[@class="stab internal"]' 0
//@ has internal_api/index.html '//dt/a[@href="struct.ExperimentalExtra.html"]/../span[@class="stab unstable"]' 'Experimental'

//@ has internal_api/struct.ExperimentalExtra.html '//div[@class="stab unstable"]//code' 'foo_internal_extra'
//@ count internal_api/struct.ExperimentalExtra.html '//div[@class="stab internal"]' 0
//@ count internal_api/struct.ExperimentalExtra.html '//div[@class="stab unstable"]' 1
#[unstable(feature = "foo_internal_extra", issue = "none")]
pub struct ExperimentalExtra;

//@ has internal_api/index.html '//dt/a[@href="struct.StableStruct.html"]' 'StableStruct'
//@ count internal_api/index.html '//dt/a[@href="struct.StableStruct.html"]/../span[@class="stab internal"]' 0
//@ count internal_api/index.html '//dt/a[@href="struct.StableStruct.html"]/../span[@class="stab unstable"]' 0

//@ has internal_api/struct.StableStruct.html
//@ count internal_api/struct.StableStruct.html '//div[@class="stab internal"]' 0
//@ count internal_api/struct.StableStruct.html '//div[@class="stab unstable"]' 0
#[stable(feature = "stable_struct", since = "1.0.0")]
pub struct StableStruct;
