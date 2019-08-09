// Various checks that stability attributes are used correctly, per RFC 507

#![feature(staged_api)]

#![stable(feature = "rust1", since = "1.0.0")]

mod bogus_attribute_types_2 {
    #[unstable] //~ ERROR malformed `unstable` attribute
    fn f1() { }

    #[unstable = "b"] //~ ERROR malformed `unstable` attribute
    fn f2() { }

    #[stable] //~ ERROR malformed `stable` attribute
    fn f3() { }

    #[stable = "a"] //~ ERROR malformed `stable` attribute
    fn f4() { }

    #[stable(feature = "a", since = "b")]
    #[rustc_deprecated] //~ ERROR malformed `rustc_deprecated` attribute
    fn f5() { }

    #[stable(feature = "a", since = "b")]
    #[rustc_deprecated = "a"] //~ ERROR malformed `rustc_deprecated` attribute
    fn f6() { }
}

fn main() { }
