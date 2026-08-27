//@ run-pass

#![feature(fn_delegation)]

// The rule is that we always generate all functions,
// even if they have a default implementation, the target expression
// (or delegation call-path) decides what to call.
mod trait_impl_to_trait {
    trait Trait {
        fn foo_default(&self) -> usize { 0 }
        fn bar_default() -> usize { 0 }
    }

    struct S;
    impl Trait for S {
        fn foo_default(&self) -> usize { 1 }
        fn bar_default() -> usize { 1 }
    }

    struct S1;
    impl Trait for S1 {
    }

    struct W(S);
    impl Trait for W {
        // All functions return `1`, as `S` overrides default implementation.
        reuse <S as Trait>::* { self.0 }
    }

    struct W1(S1);
    impl Trait for W1 {
        // All functions return `0`, as `S1` do not override implementation.
        reuse <S1 as Trait>::* { self.0 }
    }

    fn bar_default() -> usize { 2 }

    struct W2;
    impl Trait for W2 {
        // This function returns `2` as we generate delegation even if there
        // is provided default implementation.
        reuse bar_default;
    }

    struct W3(S);
    impl Trait for W3 {
        // This function returns `1` as we delegate to `S` which overrides default
        // implementation.
        reuse Trait::foo_default { self.0 }
    }

    struct W4(S1);
    impl Trait for W4 {
        // This function returns `0` as we delegate to `S1`.
        reuse Trait::foo_default { self.0 }
    }

    pub fn check() {
        assert_eq!(W(S).foo_default(), 1);
        assert_eq!(W::bar_default(), 1);

        assert_eq!(W1(S1).foo_default(), 0);
        assert_eq!(W1::bar_default(), 0);

        assert_eq!(W2::bar_default(), 2);

        assert_eq!(W3(S).foo_default(), 1);
        assert_eq!(W4(S1).foo_default(), 0);
    }
}

mod inherent_impl_to_trait {
    trait Trait {
        fn foo_default(&self) -> usize { 0 }
        fn bar_default() -> usize { 0 }
    }

    struct S;
    impl Trait for S {
        fn foo_default(&self) -> usize { 1 }
        fn bar_default() -> usize { 1 }
    }

    struct S1;
    impl Trait for S1 {
    }

    struct W(S);
    impl W {
        // Those functions return `1` as we delegate to `S`.
        reuse <S as Trait>::* { self.0 }
    }

    struct W1(S1);
    impl W1 {
        // Those functions return `0` as we delegate to `S1`.
        reuse <S1 as Trait>::* { self.0 }
    }

    pub fn check() {
        assert_eq!(W(S).foo_default(), 1);
        assert_eq!(W::bar_default(), 1);

        assert_eq!(W1(S1).foo_default(), 0);
        assert_eq!(W1::bar_default(), 0);
    }
}

fn main() {
    trait_impl_to_trait::check();
    inherent_impl_to_trait::check();
}
