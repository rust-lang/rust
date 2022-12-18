// run-rustfix

#![feature(closure_lifetime_binder)]
#![warn(clippy::explicit_auto_deref)]
#![allow(
    dead_code,
    unused_braces,
    clippy::borrowed_box,
    clippy::needless_borrow,
    clippy::needless_return,
    clippy::ptr_arg,
    clippy::redundant_field_names,
    clippy::too_many_arguments,
    clippy::borrow_deref_ref,
    clippy::let_unit_value
)]

trait CallableStr {
    type T: Fn(&str);
    fn callable_str(&self) -> Self::T;
}
impl CallableStr for () {
    type T = fn(&str);
    fn callable_str(&self) -> Self::T {
        fn f(_: &str) {}
        f
    }
}
impl CallableStr for i32 {
    type T = <() as CallableStr>::T;
    fn callable_str(&self) -> Self::T {
        ().callable_str()
    }
}

trait CallableT<U: ?Sized> {
    type T: Fn(&U);
    fn callable_t(&self) -> Self::T;
}
impl<U: ?Sized> CallableT<U> for () {
    type T = fn(&U);
    fn callable_t(&self) -> Self::T {
        fn f<U: ?Sized>(_: &U) {}
        f::<U>
    }
}
impl<U: ?Sized> CallableT<U> for i32 {
    type T = <() as CallableT<U>>::T;
    fn callable_t(&self) -> Self::T {
        ().callable_t()
    }
}

fn f_str(_: &str) {}
fn f_string(_: &String) {}
fn f_t<T>(_: T) {}
fn f_ref_t<T: ?Sized>(_: &T) {}

fn f_str_t<T>(_: &str, _: T) {}

fn f_box_t<T>(_: &Box<T>) {}

extern "C" {
    fn var(_: u32, ...);
}

fn main() {
    let s = String::new();

    let _: &str = &*s;
    let _: &str = &*{ String::new() };
    let _: &str = &mut *{ String::new() };
    let _ = &*s; // Don't lint. Inferred type would change.
    let _: &_ = &*s; // Don't lint. Inferred type would change.

    f_str(&*s);
    f_t(&*s); // Don't lint. Inferred type would change.
    f_ref_t(&*s); // Don't lint. Inferred type would change.

    f_str_t(&*s, &*s); // Don't lint second param.

    let b = Box::new(Box::new(Box::new(5)));
    let _: &Box<i32> = &**b;
    let _: &Box<_> = &**b; // Don't lint. Inferred type would change.

    f_box_t(&**b); // Don't lint. Inferred type would change.

    let c = |_x: &str| ();
    c(&*s);

    let c = |_x| ();
    c(&*s); // Don't lint. Inferred type would change.

    fn _f(x: &String) -> &str {
        &**x
    }

    fn _f1(x: &String) -> &str {
        { &**x }
    }

    fn _f2(x: &String) -> &str {
        &**{ x }
    }

    fn _f3(x: &Box<Box<Box<i32>>>) -> &Box<i32> {
        &***x
    }

    fn _f4(
        x: String,
        f1: impl Fn(&str),
        f2: &dyn Fn(&str),
        f3: fn(&str),
        f4: impl CallableStr,
        f5: <() as CallableStr>::T,
        f6: <i32 as CallableStr>::T,
        f7: &dyn CallableStr<T = fn(&str)>,
        f8: impl CallableT<str>,
        f9: <() as CallableT<str>>::T,
        f10: <i32 as CallableT<str>>::T,
        f11: &dyn CallableT<str, T = fn(&str)>,
    ) {
        f1(&*x);
        f2(&*x);
        f3(&*x);
        f4.callable_str()(&*x);
        f5(&*x);
        f6(&*x);
        f7.callable_str()(&*x);
        f8.callable_t()(&*x);
        f9(&*x);
        f10(&*x);
        f11.callable_t()(&*x);
    }

    struct S1<'a>(&'a str);
    let _ = S1(&*s);

    struct S2<'a> {
        s: &'a str,
    }
    let _ = S2 { s: &*s };

    struct S3<'a, T: ?Sized>(&'a T);
    let _ = S3(&*s); // Don't lint. Inferred type would change.

    struct S4<'a, T: ?Sized> {
        s: &'a T,
    }
    let _ = S4 { s: &*s }; // Don't lint. Inferred type would change.

    enum E1<'a> {
        S1(&'a str),
        S2 { s: &'a str },
    }
    impl<'a> E1<'a> {
        fn m1(s: &'a String) {
            let _ = Self::S1(&**s);
            let _ = Self::S2 { s: &**s };
        }
    }
    let _ = E1::S1(&*s);
    let _ = E1::S2 { s: &*s };

    enum E2<'a, T: ?Sized> {
        S1(&'a T),
        S2 { s: &'a T },
    }
    let _ = E2::S1(&*s); // Don't lint. Inferred type would change.
    let _ = E2::S2 { s: &*s }; // Don't lint. Inferred type would change.

    let ref_s = &s;
    let _: &String = &*ref_s; // Don't lint reborrow.
    f_string(&*ref_s); // Don't lint reborrow.

    struct S5 {
        foo: u32,
    }
    let b = Box::new(Box::new(S5 { foo: 5 }));
    let _ = b.foo;
    let _ = (*b).foo;
    let _ = (**b).foo;

    struct S6 {
        foo: S5,
    }
    impl core::ops::Deref for S6 {
        type Target = S5;
        fn deref(&self) -> &Self::Target {
            &self.foo
        }
    }
    let s6 = S6 { foo: S5 { foo: 5 } };
    let _ = (*s6).foo; // Don't lint. `S6` also has a field named `foo`

    let ref_str = &"foo";
    let _ = f_str(*ref_str);
    let ref_ref_str = &ref_str;
    let _ = f_str(**ref_ref_str);

    fn _f5(x: &u32) -> u32 {
        if true {
            *x
        } else {
            return *x;
        }
    }

    f_str(&&*ref_str); // `needless_borrow` will suggest removing both references
    f_str(&&**ref_str); // `needless_borrow` will suggest removing only one reference

    let x = &&40;
    unsafe {
        var(0, &**x);
    }

    let s = &"str";
    let _ = || return *s;
    let _ = || -> &'static str { return *s };

    struct X;
    struct Y(X);
    impl core::ops::Deref for Y {
        type Target = X;
        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }
    let _: &X = &*{ Y(X) };
    let _: &X = &*match 0 {
        #[rustfmt::skip]
        0 => { Y(X) },
        _ => panic!(),
    };
    let _: &X = &*if true { Y(X) } else { panic!() };

    fn deref_to_u<U, T: core::ops::Deref<Target = U>>(x: &T) -> &U {
        &**x
    }

    let _ = |x: &'static Box<dyn Iterator<Item = u32>>| -> &'static dyn Iterator<Item = u32> { &**x };
    fn ret_any(x: &Box<dyn std::any::Any>) -> &dyn std::any::Any {
        &**x
    }

    let x = String::new();
    let _: *const str = &*x;

    struct S7([u32; 1]);
    impl core::ops::Deref for S7 {
        type Target = [u32; 1];
        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }
    let x = S7([0]);
    let _: &[u32] = &*x;

    let c1 = |_: &Vec<&u32>| {};
    let x = &&vec![&1u32];
    c1(*x);
    let _ = for<'a, 'b> |x: &'a &'a Vec<&'b u32>, b: bool| -> &'a Vec<&'b u32> {
        if b {
            return *x;
        }
        *x
    };

    trait WithAssoc {
        type Assoc: ?Sized;
    }
    impl WithAssoc for String {
        type Assoc = str;
    }
    fn takes_assoc<T: WithAssoc>(_: &T::Assoc) -> T {
        unimplemented!()
    }
    let _: String = takes_assoc(&*String::new());

    // Issue #9901
    fn takes_ref(_: &i32) {}
    takes_ref(*Box::new(&0i32));
}
