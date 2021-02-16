// Various checks that stability attributes are used correctly, per RFC 507

#![feature(const_fn, staged_api)]

#![stable(feature = "rust1", since = "1.0.0")]

mod bogus_attribute_types_1 {
    #[stable(feature = "a", since = "1.0.0", reason)] //~ ERROR unknown meta item 'reason' [E0541]
    fn f1() { }

    #[stable(feature = "a", since)] //~ ERROR incorrect meta item [E0539]
    fn f2() { }

    #[stable(feature, since = "1.0.0")] //~ ERROR incorrect meta item [E0539]
    fn f3() { }

    #[stable(feature = "a", since(b))] //~ ERROR incorrect meta item [E0539]
    fn f5() { }

    #[stable(feature(b), since = "1.0.0")] //~ ERROR incorrect meta item [E0539]
    fn f6() { }
}

mod missing_feature_names {
    #[unstable(issue = "none")] //~ ERROR missing 'feature' [E0546]
    fn f1() { }

    #[unstable(feature = "b")] //~ ERROR missing 'issue' [E0547]
    fn f2() { }

    #[stable(since = "1.0.0")] //~ ERROR missing 'feature' [E0546]
    fn f3() { }
}

mod missing_version {
    #[stable(feature = "a")] //~ ERROR invalid 'since' [E0542]
    fn f1() { }

    #[stable(feature = "a", since = "1.0.0")]
    #[rustc_deprecated(reason = "a")] //~ ERROR invalid 'since' [E0542]
    fn f2() { }

    #[stable(feature = "a", since = "1.0.0")]
    #[rustc_deprecated(since = "1.0.0")] //~ ERROR missing 'reason' [E0543]
    fn f3() { }
}

#[unstable(feature = "b", issue = "none")]
#[stable(feature = "a", since = "1.0.0")] //~ ERROR multiple stability levels [E0544]
fn multiple1() { }

#[unstable(feature = "b", issue = "none")]
#[unstable(feature = "b", issue = "none")] //~ ERROR multiple stability levels [E0544]
fn multiple2() { }

#[stable(feature = "a", since = "1.0.0")]
#[stable(feature = "a", since = "1.0.0")] //~ ERROR multiple stability levels [E0544]
fn multiple3() { }

#[stable(feature = "a", since = "1.0.0")]
#[rustc_deprecated(since = "1.0.0", reason = "text")]
#[rustc_deprecated(since = "1.0.0", reason = "text")] //~ ERROR multiple deprecated attributes
#[rustc_const_unstable(feature = "c", issue = "none")]
#[rustc_const_unstable(feature = "d", issue = "none")] //~ ERROR multiple stability levels
pub const fn multiple4() { }

#[stable(feature = "a", since = "invalid")] //~ ERROR invalid 'since' [E0542]
fn invalid_stability_version() {}

#[stable(feature = "a", since = "1.0.0")]
#[rustc_deprecated(since = "invalid", reason = "text")] //~ ERROR invalid 'since' [E0542]
fn invalid_deprecation_version() {}

#[rustc_deprecated(since = "1.0.0", reason = "text")]
fn deprecated_without_unstable_or_stable() { }
//~^^ ERROR rustc_deprecated attribute must be paired with either stable or unstable attribute

#[stable(feature = "a", since = "2.0.0")]
#[rustc_deprecated(since = "1.0.0", reason = "text")]
fn deprecated_before_stabilized() {} //~ ERROR An API can't be stabilized after it is deprecated

fn main() { }
