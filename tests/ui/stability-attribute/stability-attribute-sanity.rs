// Various checks that stability attributes are used correctly, per RFC 507

#![feature(staged_api)]

#![stable(feature = "rust1", since = "1.0.0")]

mod bogus_attribute_types_1 {
    #[stable(feature = "a", since = "4.4.4", reason)] //~ ERROR unknown meta item 'reason' [E0541]
    fn f1() { }

    #[stable(feature = "a", since)] //~ ERROR malformed `stable` attribute input [E0539]
    fn f2() { }

    #[stable(feature, since = "3.3.3")] //~ ERROR malformed `stable` attribute input [E0539]
    fn f3() { }

    #[stable(feature = "a", since(b))] //~ ERROR malformed `stable` attribute input [E0539]
    fn f5() { }

    #[stable(feature(b), since = "3.3.3")] //~ ERROR malformed `stable` attribute input [E0539]
    fn f6() { }
}

mod missing_feature_names {
    #[unstable(issue = "none")] //~ ERROR missing 'feature' [E0546]
    fn f1() { }

    #[unstable(feature = "b")] //~ ERROR missing 'issue' [E0547]
    fn f2() { }

    #[stable(since = "3.3.3")] //~ ERROR missing 'feature' [E0546]
    fn f3() { }
}

mod missing_version {
    #[stable(feature = "a")] //~ ERROR missing 'since' [E0542]
    fn f1() { }

    #[stable(feature = "a", since = "4.4.4")]
    #[deprecated(note = "a")] //~ ERROR missing 'since' [E0542]
    fn f2() { }

    #[stable(feature = "a", since = "4.4.4")]
    #[deprecated(since = "5.5.5")] //~ ERROR missing 'note' [E0543]
    fn f3() { }
}

#[unstable(feature = "b", issue = "none")]
#[stable(feature = "a", since = "4.4.4")] //~ ERROR multiple stability levels [E0544]
fn multiple1() { }

#[unstable(feature = "b", issue = "none")]
#[unstable(feature = "b", issue = "none")] //~ ERROR multiple stability levels [E0544]
fn multiple2() { }

#[stable(feature = "a", since = "4.4.4")]
#[stable(feature = "a", since = "4.4.4")] //~ ERROR multiple stability levels [E0544]
fn multiple3() { }

#[stable(feature = "e", since = "b")] //~ ERROR 'since' must be a Rust version number, such as "1.31.0"
#[deprecated(since = "5.5.5", note = "text")]
#[deprecated(since = "5.5.5", note = "text")] //~ ERROR multiple `deprecated` attributes
#[rustc_const_unstable(feature = "c", issue = "none")]
#[rustc_const_unstable(feature = "d", issue = "none")] //~ ERROR multiple stability levels
pub const fn multiple4() { }

#[stable(feature = "a", since = "1.0.0")] //~ ERROR feature `a` is declared stable since 1.0.0
#[deprecated(since = "invalid", note = "text")] //~ ERROR 'since' must be a Rust version number, such as "1.31.0"
fn invalid_deprecation_version() {}

#[deprecated(since = "5.5.5", note = "text")]
fn deprecated_without_unstable_or_stable() { }
//~^^ ERROR deprecated attribute must be paired with either stable or unstable attribute

fn main() { }
