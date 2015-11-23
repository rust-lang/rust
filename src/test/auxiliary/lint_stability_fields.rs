// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(staged_api)]
#![staged_api]
#![stable(feature = "rust1", since = "1.0.0")]

#[stable(feature = "rust1", since = "1.0.0")]
pub struct Stable {
    #[stable(feature = "rust1", since = "1.0.0")]
    pub inherit: u8, // it's a lie (stable doesn't inherit)
    #[unstable(feature = "test_feature", issue = "0")]
    pub override1: u8,
    #[rustc_deprecated(since = "1.0.0", reason = "text")]
    #[unstable(feature = "test_feature", issue = "0")]
    pub override2: u8,
}

#[stable(feature = "rust1", since = "1.0.0")]
pub struct Stable2(#[stable(feature = "rust1", since = "1.0.0")] pub u8,
                   #[unstable(feature = "test_feature", issue = "0")] pub u8,
                   #[unstable(feature = "test_feature", issue = "0")]
                   #[rustc_deprecated(since = "1.0.0", reason = "text")] pub u8);

#[unstable(feature = "test_feature", issue = "0")]
pub struct Unstable {
    pub inherit: u8,
    #[stable(feature = "rust1", since = "1.0.0")]
    pub override1: u8,
    #[rustc_deprecated(since = "1.0.0", reason = "text")]
    #[unstable(feature = "test_feature", issue = "0")]
    pub override2: u8,
}

#[unstable(feature = "test_feature", issue = "0")]
pub struct Unstable2(pub u8,
                     #[stable(feature = "rust1", since = "1.0.0")] pub u8,
                     #[unstable(feature = "test_feature", issue = "0")]
                     #[rustc_deprecated(since = "1.0.0", reason = "text")] pub u8);

#[unstable(feature = "test_feature", issue = "0")]
#[rustc_deprecated(since = "1.0.0", reason = "text")]
pub struct Deprecated {
    pub inherit: u8,
    #[stable(feature = "rust1", since = "1.0.0")]
    pub override1: u8,
    #[unstable(feature = "test_feature", issue = "0")]
    pub override2: u8,
}

#[unstable(feature = "test_feature", issue = "0")]
#[rustc_deprecated(since = "1.0.0", reason = "text")]
pub struct Deprecated2(pub u8,
                       #[stable(feature = "rust1", since = "1.0.0")] pub u8,
                       #[unstable(feature = "test_feature", issue = "0")] pub u8);
