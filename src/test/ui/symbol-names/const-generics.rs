// check-pass
// revisions: legacy v0
//[legacy]compile-flags: -Z symbol-mangling-version=legacy --crate-type=lib
    //[v0]compile-flags: -Z symbol-mangling-version=v0 --crate-type=lib

    #![feature(min_const_generics)]

    // `char`
    pub struct Char<const F: char>;

    impl Char<'A'> {
        pub fn foo() {}
    }

    impl<const F: char> Char<F> {
        pub fn bar() {}
    }

    // `i8`
    pub struct I8<const F: i8>;

    impl I8<{std::i8::MIN}> {
        pub fn foo() {}
    }

    impl I8<{std::i8::MAX}> {
        pub fn foo() {}
    }

    impl<const F: i8> I8<F> {
        pub fn bar() {}
    }

    // `i16`
    pub struct I16<const F: i16>;

    impl I16<{std::i16::MIN}> {
        pub fn foo() {}
    }

    impl<const F: i16> I16<F> {
        pub fn bar() {}
    }

    // `i32`
    pub struct I32<const F: i32>;

    impl I32<{std::i32::MIN}> {
        pub fn foo() {}
    }

    impl<const F: i32> I32<F> {
        pub fn bar() {}
    }

    // `i64`
    pub struct I64<const F: i64>;

    impl I64<{std::i64::MIN}> {
        pub fn foo() {}
    }

    impl<const F: i64> I64<F> {
        pub fn bar() {}
    }

    // `i128`
    pub struct I128<const F: i128>;

    impl I128<{std::i128::MIN}> {
        pub fn foo() {}
    }

    impl<const F: i128> I128<F> {
        pub fn bar() {}
    }

    // `isize`
    pub struct ISize<const F: isize>;

    impl ISize<3> {
        pub fn foo() {}
    }

    impl<const F: isize> ISize<F> {
        pub fn bar() {}
    }
