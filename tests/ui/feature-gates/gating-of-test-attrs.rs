#![feature(test)]

// test is a built-in macro, not a built-in attribute, but it kind of acts like both.
// check its target checking anyway here
#[test]
//~^ ERROR the `#[test]` attribute may only be used on a free function
mod test {
    mod inner { #![test] }
    //~^ ERROR inner macro attributes are unstable
    //~| ERROR the `#[test]` attribute may only be used on a free function

    #[test]
    //~^ ERROR the `#[test]` attribute may only be used on a free function
    struct S;

    #[test]
    //~^ ERROR the `#[test]` attribute may only be used on a free function
    type T = S;

    #[test]
    //~^ ERROR the `#[test]` attribute may only be used on a free function
    impl S { }
}

// At time of unit test authorship, if compiling without `--test` then
// non-crate-level #[bench] attributes seem to be ignored.

#[bench]
//~^ ERROR the `#[bench]` attribute may only be used on a free function
mod bench {
    mod inner { #![bench] }
    //~^ ERROR inner macro attributes are unstable
    //~| ERROR the `#[bench]` attribute may only be used on a free function

    #[bench]
    //~^ ERROR the `#[bench]` attribute may only be used on a free function
    struct S;

    #[bench]
    //~^ ERROR the `#[bench]` attribute may only be used on a free function
    type T = S;

    #[bench]
    //~^ ERROR the `#[bench]` attribute may only be used on a free function
    impl S { }
}

fn main() {}
