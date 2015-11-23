// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Various checks that stability attributes are used correctly, per RFC 507

#![feature(staged_api)]
#![staged_api]

#![stable(feature = "rust1", since = "1.0.0")]

mod bogus_attribute_types_1 {
    #[stable(feature = "a", since = "a", reason)] //~ ERROR unknown meta item 'reason'
    fn f1() { }

    #[stable(feature = "a", since)] //~ ERROR incorrect meta item
    fn f2() { }

    #[stable(feature, since = "a")] //~ ERROR incorrect meta item
    fn f3() { }

    #[stable(feature = "a", since(b))] //~ ERROR incorrect meta item
    fn f5() { }

    #[stable(feature(b), since = "a")] //~ ERROR incorrect meta item
    fn f6() { }
}

mod bogus_attribute_types_2 {
    #[unstable] //~ ERROR incorrect stability attribute type
    fn f1() { }

    #[unstable = "a"] //~ ERROR incorrect stability attribute type
    fn f2() { }

    #[stable] //~ ERROR incorrect stability attribute type
    fn f3() { }

    #[stable = "a"] //~ ERROR incorrect stability attribute type
    fn f4() { }

    #[stable(feature = "a", since = "b")]
    #[rustc_deprecated] //~ ERROR incorrect stability attribute type
    fn f5() { }

    #[stable(feature = "a", since = "b")]
    #[rustc_deprecated = "a"] //~ ERROR incorrect stability attribute type
    fn f6() { }
}

mod missing_feature_names {
    #[unstable(issue = "0")] //~ ERROR missing 'feature'
    fn f1() { }

    #[unstable(feature = "a")] //~ ERROR missing 'issue'
    fn f2() { }

    #[stable(since = "a")] //~ ERROR missing 'feature'
    fn f3() { }
}

mod missing_version {
    #[stable(feature = "a")] //~ ERROR missing 'since'
    fn f1() { }

    #[stable(feature = "a", since = "b")]
    #[rustc_deprecated(reason = "a")] //~ ERROR missing 'since'
    fn f2() { }
}

#[unstable(feature = "a", issue = "0")]
#[stable(feature = "a", since = "b")]
fn multiple1() { } //~ ERROR multiple stability levels

#[unstable(feature = "a", issue = "0")]
#[unstable(feature = "a", issue = "0")]
fn multiple2() { } //~ ERROR multiple stability levels

#[stable(feature = "a", since = "b")]
#[stable(feature = "a", since = "b")]
fn multiple3() { } //~ ERROR multiple stability levels

#[stable(feature = "a", since = "b")]
#[rustc_deprecated(since = "b", reason = "text")]
#[rustc_deprecated(since = "b", reason = "text")]
fn multiple4() { } //~ ERROR multiple rustc_deprecated attributes
//~^ ERROR Invalid stability or deprecation version found

#[rustc_deprecated(since = "a", reason = "text")]
fn deprecated_without_unstable_or_stable() { } //~ ERROR rustc_deprecated attribute must be paired

fn main() { }
