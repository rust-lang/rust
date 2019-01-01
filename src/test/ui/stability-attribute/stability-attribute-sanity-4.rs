// Various checks that stability attributes are used correctly, per RFC 507

#![feature(staged_api)]

#![stable(feature = "rust1", since = "1.0.0")]

mod bogus_attribute_types_2 {
    #[unstable] //~ ERROR attribute must be of the form
    fn f1() { }

    #[unstable = "b"] //~ ERROR attribute must be of the form
    fn f2() { }

    #[stable] //~ ERROR attribute must be of the form
    fn f3() { }

    #[stable = "a"] //~ ERROR attribute must be of the form
    fn f4() { }

    #[stable(feature = "a", since = "b")]
    #[rustc_deprecated] //~ ERROR attribute must be of the form
    fn f5() { }

    #[stable(feature = "a", since = "b")]
    #[rustc_deprecated = "a"] //~ ERROR attribute must be of the form
    fn f6() { }
}

fn main() { }
