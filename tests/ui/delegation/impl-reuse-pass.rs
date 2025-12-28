//@ check-pass

#![allow(incomplete_features)]
#![feature(fn_delegation)]
#![feature(const_trait_impl)]
#![allow(warnings)]

mod default {
    trait T {
        fn foo(&self) {}
        fn bar(&self) {}
        fn goo(&self) {}
    }

    struct S;
    impl T for S {}

    struct F(S);
    reuse impl T for F { self.0 }

    fn f() {
        let f = F(S{});

        f.foo();
        f.bar();
        f.goo();
    }
}

mod dyn_traits {
    trait T {
        fn foo(&self) -> Box<dyn T>;
    }

    trait SecondTrait {
        fn bar(&self);
    }

    reuse impl SecondTrait for dyn T { self.foo().as_ref() }
}

mod complex_path {
    pub mod first {
        pub mod second {
            pub trait T {
                fn foo(&self, x: usize);
            }
        }
    }

    struct S;
    impl first::second::T for S {
        fn foo(&self, x: usize) { }
    }

    struct F(S);
    reuse impl first::second::T for F { self.0 }

    fn f() {
        use complex_path::first::second::T;

        let f = F(S{});

        f.foo(1);
    }
}

mod no_body_reuse {
    trait T {
        fn foo(&self) {}
        fn bar(&mut self) {}
    }

    struct F;

    reuse impl T for F;

    fn foo() {
        let mut f = F{};

        f.foo();
        f.bar();
    }
}

mod unsafe_trait {
    unsafe trait UnsafeTrait {
        fn foo(&self) {}
        fn bar(&self) {}
        fn goo(&self) {}
    }

    struct S;
    unsafe impl UnsafeTrait for S {}

    struct F(S);
    reuse unsafe impl UnsafeTrait for F { self.0 }

    fn f() {
        let f = F(S{});

        f.foo();
        f.bar();
        f.goo();
    }
}

mod const_trait {
    const trait ConstTrait {
        fn foo(&self) -> usize { 0 }
        fn bar(&self) -> usize { 1 }
    }

    struct S;
    const impl ConstTrait for S {}

    struct F(S);
    reuse const impl ConstTrait for F { self.0 }

    fn f() {
        let f = F(S{});

        f.foo();
        f.bar();
    }
}

mod different_selves {
    trait T: Sized {
        fn foo(&self) {}
        fn boo(self) {}
        fn goo(&mut self) {}
    }

    struct S;
    impl T for S {}

    struct F(S);
    reuse impl T for F { self.0 }

    struct D(S);
    macro_rules! self_0 { ($self:ident) => { $self.0 } }

    reuse impl T for D { self_0!(self) }

    fn f() {
        let mut f = F(S{});
        f.foo();
        f.goo();
        f.boo();

        let mut d = D(S{});
        d.foo();
        d.goo();
        d.boo();
    }
}

mod macros {
    trait Trait {
        fn foo(&self) -> u8 { 0 }
        fn bar(&self) -> u8 { 1 }
    }

    impl Trait for u8 {}
    struct S(u8);

    macro_rules! self_0_ref { ($self:ident) => { &$self.0 } }

    reuse impl Trait for S { self_0_ref!(self) }

    struct M(u8);
    macro_rules! m { () => { M } }
    reuse impl Trait for m!() { self_0_ref!(self) }

    struct S1(u8);
    macro_rules! one_line_reuse { ($self:ident) => { reuse impl Trait for S1 { $self.0 } } }
    one_line_reuse!(self);

    struct S2(u8);
    macro_rules! one_line_reuse_expr { ($x:expr) => { reuse impl Trait for S2 { $x } } }
    one_line_reuse_expr!(self.0);

    struct S3(u8);
    macro_rules! s3 { () => { S3 } }
    macro_rules! one_line_reuse_expr2 { ($x:expr) => { reuse impl Trait for s3!() { $x } } }
    one_line_reuse_expr2!(self.0);

    fn f() {
        let s = S(1);
        s.foo();
        s.bar();

        let m = M(41);
        m.foo();
        m.bar();

        let s1 = S1(2);
        s1.foo();
        s1.bar();

        let s2 = S2(4);
        s2.foo();
        s2.bar();

        let s3 = S3(5);
        s3.foo();
        s3.bar();
    }
}

mod generics {
    trait Trait<'a, 'b, A, B, C> {
        fn foo(&self, a: &A) {}
        fn bar(&self, b: &B) {}
        fn goo(&self, c: &C) {}
    }

    struct S;
    impl<'a, 'b, A, B, C> Trait<'a, 'b, A, B, C> for S {}

    struct F(S);
    reuse impl<'a, 'b, A, B, C> Trait<'a, 'b, A, B, C> for F { &self.0 }

    struct S1;
    struct F1(S1);
    impl<'c, B> Trait<'static, 'c, usize, B, String> for S1 {}
    reuse impl<'d, B> Trait<'static, 'd, usize, B, String> for F1 { &self.0 }

    struct S2;
    struct F2(S2);
    impl Trait<'static, 'static, u8, u16, u32> for S2 {}
    reuse impl Trait<'static, 'static, u8, u16, u32> for F2 { &self.0 }

    fn f<'a, 'b, 'c, A, B, C>(a: A, b: B, c: C) {
        let f = F(S{});

        <F as Trait<'a, 'b, A, B, C>>::foo(&f, &a);
        <F as Trait<'a, 'b, A, B, C>>::bar(&f, &b);
        <F as Trait<'a, 'b, A, B, C>>::goo(&f, &c);

        let f = F1(S1{});
        <F1 as Trait<'static, 'c, usize, B, String>>::foo(&f, &123);
        <F1 as Trait<'static, 'c, usize, B, String>>::bar(&f, &b);
        <F1 as Trait<'static, 'c, usize, B, String>>::goo(&f, &"s".to_string());

        let f = F2(S2{});
        <F2 as Trait<'static, 'static, u8, u16, u32>>::foo(&f, &1);
        <F2 as Trait<'static, 'static, u8, u16, u32>>::bar(&f, &2);
        <F2 as Trait<'static, 'static, u8, u16, u32>>::goo(&f, &3);
    }
}

mod reuse_in_different_places {
    trait T {
        fn foo(&self, x: usize) {}
    }

    struct S;
    impl T for S {}

    struct F1(S);
    reuse impl T for F1 {
        struct F2(S, S, S);
        reuse impl T for F2 { self.1 }

        let f2 = F2(S{}, S{}, S{});
        f2.foo(123);

        &self.0
    }

    fn foo() {
        struct F(S);
        reuse impl T for F { self.0 }

        let f = F(S{});
        f.foo(1);
    }

    fn bar() {
        || {
            struct F(S);
            reuse impl T for F { self.0 }

            let f = F(S{});
            f.foo(1);
        };
    }
}

fn main() {}
