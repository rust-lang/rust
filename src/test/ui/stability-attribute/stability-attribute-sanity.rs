// Various checks that stability attributes are used correctly, per RFC 507

#![feature(const_fn, staged_api)]

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
    #[unstable(issue = "none")] //~ ERROR missing 'feature' [E0546]
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

    #[stable(feature = "a", since = "b")]
    #[rustc_deprecated(since = "a")] //~ ERROR missing 'reason' [E0543]
    fn f3() { }
}

#[unstable(feature = "b", issue = "none")]
#[stable(feature = "a", since = "b")] //~ ERROR multiple stability levels [E0544]
fn multiple1() { }

#[unstable(feature = "b", issue = "none")]
#[unstable(feature = "b", issue = "none")] //~ ERROR multiple stability levels [E0544]
fn multiple2() { }

#[stable(feature = "a", since = "b")]
#[stable(feature = "a", since = "b")] //~ ERROR multiple stability levels [E0544]
fn multiple3() { }

#[stable(feature = "a", since = "b")]
#[rustc_deprecated(since = "b", reason = "text")]
#[rustc_deprecated(since = "b", reason = "text")] //~ ERROR multiple deprecated attributes
#[rustc_const_unstable(feature = "c", issue = "none")]
#[rustc_const_unstable(feature = "d", issue = "none")] //~ ERROR multiple stability levels
pub const fn multiple4() { }
//~^ ERROR Invalid stability version found

#[stable(feature = "a", since = "1.0.0")]
#[rustc_deprecated(since = "invalid", reason = "text")]
fn invalid_deprecation_version() {} //~ ERROR Invalid deprecation version found

#[rustc_deprecated(since = "a", reason = "text")]
fn deprecated_without_unstable_or_stable() { }
//~^^ ERROR rustc_deprecated attribute must be paired with either stable or unstable attribute

fn main() { }
