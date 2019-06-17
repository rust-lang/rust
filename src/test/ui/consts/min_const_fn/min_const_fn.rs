// ok
const fn foo1() {}
const fn foo2(x: i32) -> i32 { x }
const fn foo3<T>(x: T) -> T { x }
const fn foo7() {
    (
        foo1(),
        foo2(420),
        foo3(69),
    ).0
}
const fn foo12<T: Sized>(t: T) -> T { t }
const fn foo13<T: ?Sized>(t: &T) -> &T { t }
const fn foo14<'a, T: 'a>(t: &'a T) -> &'a T { t }
const fn foo15<T>(t: T) -> T where T: Sized { t }
const fn foo15_2<T>(t: &T) -> &T where T: ?Sized { t }
const fn foo16(f: f32) -> f32 { f }
const fn foo17(f: f32) -> u32 { f as u32 }
const fn foo18(i: i32) -> i32 { i * 3 }
const fn foo20(b: bool) -> bool { !b }
const fn foo21<T, U>(t: T, u: U) -> (T, U) { (t, u) }
const fn foo22(s: &[u8], i: usize) -> u8 { s[i] }
const FOO: u32 = 42;
const fn foo23() -> u32 { FOO }
const fn foo24() -> &'static u32 { &FOO }
const fn foo27(x: &u32) -> u32 { *x }
const fn foo28(x: u32) -> u32 { *&x }
const fn foo29(x: u32) -> i32 { x as i32 }
const fn foo31(a: bool, b: bool) -> bool { a & b }
const fn foo32(a: bool, b: bool) -> bool { a | b }
const fn foo33(a: bool, b: bool) -> bool { a & b }
const fn foo34(a: bool, b: bool) -> bool { a | b }
const fn foo35(a: bool, b: bool) -> bool { a ^ b }
struct Foo<T: ?Sized>(T);
impl<T> Foo<T> {
    const fn new(t: T) -> Self { Foo(t) }
    const fn into_inner(self) -> T { self.0 } //~ destructors cannot be evaluated
    const fn get(&self) -> &T { &self.0 }
    const fn get_mut(&mut self) -> &mut T { &mut self.0 }
    //~^ mutable references in const fn are unstable
}
impl<'a, T> Foo<T> {
    const fn new_lt(t: T) -> Self { Foo(t) }
    const fn into_inner_lt(self) -> T { self.0 } //~ destructors cannot be evaluated
    const fn get_lt(&'a self) -> &T { &self.0 }
    const fn get_mut_lt(&'a mut self) -> &mut T { &mut self.0 }
    //~^ mutable references in const fn are unstable
}
impl<T: Sized> Foo<T> {
    const fn new_s(t: T) -> Self { Foo(t) }
    const fn into_inner_s(self) -> T { self.0 } //~ ERROR destructors
    const fn get_s(&self) -> &T { &self.0 }
    const fn get_mut_s(&mut self) -> &mut T { &mut self.0 }
    //~^ mutable references in const fn are unstable
}
impl<T: ?Sized> Foo<T> {
    const fn get_sq(&self) -> &T { &self.0 }
    const fn get_mut_sq(&mut self) -> &mut T { &mut self.0 }
    //~^ mutable references in const fn are unstable
}


const fn char_ops(c: char, d: char) -> bool { c == d }
const fn char_ops2(c: char, d: char) -> bool { c < d }
const fn char_ops3(c: char, d: char) -> bool { c != d }
const fn i32_ops(c: i32, d: i32) -> bool { c == d }
const fn i32_ops2(c: i32, d: i32) -> bool { c < d }
const fn i32_ops3(c: i32, d: i32) -> bool { c != d }
const fn i32_ops4(c: i32, d: i32) -> i32 { c + d }
const fn char_cast(u: u8) -> char { u as char }
const unsafe fn ret_i32_no_unsafe() -> i32 { 42 }
const unsafe fn ret_null_ptr_no_unsafe<T>() -> *const T { core::ptr::null() }
const unsafe fn ret_null_mut_ptr_no_unsafe<T>() -> *mut T { core::ptr::null_mut() }

// not ok
const fn foo11<T: std::fmt::Display>(t: T) -> T { t }
//~^ ERROR trait bounds other than `Sized` on const fn parameters are unstable
const fn foo11_2<T: Send>(t: T) -> T { t }
//~^ ERROR trait bounds other than `Sized` on const fn parameters are unstable
const fn foo19(f: f32) -> f32 { f * 2.0 }
//~^ ERROR only int, `bool` and `char` operations are stable in const fn
const fn foo19_2(f: f32) -> f32 { 2.0 - f }
//~^ ERROR only int, `bool` and `char` operations are stable in const fn
const fn foo19_3(f: f32) -> f32 { -f }
//~^ ERROR only int and `bool` operations are stable in const fn
const fn foo19_4(f: f32, g: f32) -> f32 { f / g }
//~^ ERROR only int, `bool` and `char` operations are stable in const fn

static BAR: u32 = 42;
const fn foo25() -> u32 { BAR } //~ ERROR cannot access `static` items in const fn
const fn foo26() -> &'static u32 { &BAR } //~ ERROR cannot access `static` items
const fn foo30(x: *const u32) -> usize { x as usize }
//~^ ERROR casting pointers to ints is unstable
const fn foo30_with_unsafe(x: *const u32) -> usize { unsafe { x as usize } }
//~^ ERROR casting pointers to ints is unstable
const fn foo30_2(x: *mut u32) -> usize { x as usize }
//~^ ERROR casting pointers to ints is unstable
const fn foo30_2_with_unsafe(x: *mut u32) -> usize { unsafe { x as usize } }
//~^ ERROR casting pointers to ints is unstable
const fn foo30_4(b: bool) -> usize { if b { 1 } else { 42 } }
//~^ ERROR loops and conditional expressions are not stable in const fn
const fn foo30_5(b: bool) { while b { } } //~ ERROR not stable in const fn
const fn foo30_6() -> bool { let x = true; x }
const fn foo36(a: bool, b: bool) -> bool { a && b }
//~^ ERROR loops and conditional expressions are not stable in const fn
const fn foo37(a: bool, b: bool) -> bool { a || b }
//~^ ERROR loops and conditional expressions are not stable in const fn
const fn inc(x: &mut i32) { *x += 1 }
//~^ ERROR mutable references in const fn are unstable

fn main() {}

impl<T: std::fmt::Debug> Foo<T> {
//~^ ERROR trait bounds other than `Sized` on const fn parameters are unstable
    const fn foo(&self) {}
}

impl<T: std::fmt::Debug + Sized> Foo<T> {
//~^ ERROR trait bounds other than `Sized` on const fn parameters are unstable
    const fn foo2(&self) {}
}

impl<T: Sync + Sized> Foo<T> {
//~^ ERROR trait bounds other than `Sized` on const fn parameters are unstable
    const fn foo3(&self) {}
}

struct AlanTuring<T>(T);
const fn no_rpit2() -> AlanTuring<impl std::fmt::Debug> { AlanTuring(0) }
//~^ ERROR `impl Trait` in const fn is unstable
const fn no_apit2(_x: AlanTuring<impl std::fmt::Debug>) {}
//~^ ERROR trait bounds other than `Sized`
const fn no_apit(_x: impl std::fmt::Debug) {} //~ ERROR trait bounds other than `Sized`
const fn no_rpit() -> impl std::fmt::Debug {} //~ ERROR `impl Trait` in const fn is unstable
const fn no_dyn_trait(_x: &dyn std::fmt::Debug) {} //~ ERROR trait bounds other than `Sized`
const fn no_dyn_trait_ret() -> &'static dyn std::fmt::Debug { &() }
//~^ ERROR trait bounds other than `Sized`
//~| WARNING cannot return reference to temporary value
//~| WARNING this error has been downgraded to a warning
//~| WARNING this warning will become a hard error in the future

const fn no_unsafe() { unsafe {} }

const fn really_no_traits_i_mean_it() { (&() as &dyn std::fmt::Debug, ()).1 }
//~^ ERROR trait bounds other than `Sized`

const fn no_fn_ptrs(_x: fn()) {}
//~^ ERROR function pointers in const fn are unstable
const fn no_fn_ptrs2() -> fn() { fn foo() {} foo }
//~^ ERROR function pointers in const fn are unstable
