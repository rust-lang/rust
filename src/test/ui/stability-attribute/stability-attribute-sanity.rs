// Various checks that stability attributes are used correctly, per RFC 507

#![feature(const_fn, staged_api, rustc_const_unstable)]

#![stable(feature = "rust1", since = "1.0.0")]

mod bogus_attribute_types_1 {
    #[stable(feature = "a", since = "b", reason)] //~ ERROR unknown meta item 'reason' [E0541]
    fn f1() { }

    #[stable(feature = "a", since)] //~ ERROR incorrect meta item [E0539]
    fn f2() { }

    #[stable(feature, since = "a")] //~ ERROR incorrect meta item [E0539]
    fn f3() { }

    #[stable(feature = "a", since(b))] //~ ERROR incorrect meta item [E0539]
    fn f5() { }

    #[stable(feature(b), since = "a")] //~ ERROR incorrect meta item [E0539]
    fn f6() { }
}

mod missing_feature_names {
    #[unstable(issue = "0")] //~ ERROR missing 'feature' [E0546]
    fn f1() { }

    #[unstable(feature = "b")] //~ ERROR missing 'issue' [E0547]
    fn f2() { }

    #[stable(since = "a")] //~ ERROR missing 'feature' [E0546]
    fn f3() { }
}

mod missing_version {
    #[stable(feature = "a")] //~ ERROR missing 'since' [E0542]
    fn f1() { }

    #[stable(feature = "a", since = "b")]
    #[rustc_deprecated(reason = "a")] //~ ERROR missing 'since' [E0542]
    fn f2() { }
}

#[unstable(feature = "b", issue = "0")]
#[stable(feature = "a", since = "b")] //~ ERROR multiple stability levels [E0544]
fn multiple1() { }

#[unstable(feature = "b", issue = "0")]
#[unstable(feature = "b", issue = "0")] //~ ERROR multiple stability levels [E0544]
fn multiple2() { }

#[stable(feature = "a", since = "b")]
#[stable(feature = "a", since = "b")] //~ ERROR multiple stability levels [E0544]
fn multiple3() { }

#[stable(feature = "a", since = "b")]
#[rustc_deprecated(since = "b", reason = "text")]
#[rustc_deprecated(since = "b", reason = "text")]
#[rustc_const_unstable(feature = "c")]
#[rustc_const_unstable(feature = "d")]
pub const fn multiple4() { } //~ ERROR multiple rustc_deprecated attributes [E0540]
//~^ ERROR Invalid stability or deprecation version found
//~| ERROR multiple rustc_const_unstable attributes

#[rustc_deprecated(since = "a", reason = "text")]
fn deprecated_without_unstable_or_stable() { }
//~^ ERROR rustc_deprecated attribute must be paired with either stable or unstable attribute

fn main() { }
