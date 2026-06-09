//@ run-rustfix

#![feature(impl_trait_in_fn_trait_return)]
#![deny(unused_parens)]
#![allow(while_true)] // for rustfix

#[derive(Eq, PartialEq)]
struct X { y: bool }
impl X {
    fn foo(&self, conjunct: bool) -> bool { self.y && conjunct }
}

fn foo() -> isize {
    return (1); //~ ERROR unnecessary parentheses around `return` value
}
fn bar(y: bool) -> X {
    return (X { y }); //~ ERROR unnecessary parentheses around `return` value
}

pub fn around_return_type() -> (u32) { //~ ERROR unnecessary parentheses around type
    panic!()
}

pub fn around_block_return() -> u32 {
    let _foo = {
        (5) //~ ERROR unnecessary parentheses around block return value
    };
    (5) //~ ERROR unnecessary parentheses around block return value
}

pub trait Trait {
    fn test(&self);
}

pub fn around_multi_bound_ref() -> &'static (dyn Trait + Send) {
    panic!()
}

//~v ERROR unnecessary parentheses around type
pub fn around_single_bound_ref() -> &'static (dyn Trait) {
    panic!()
}

pub fn around_multi_bound_ptr() -> *const (dyn Trait + Send) {
    panic!()
}

//~v ERROR unnecessary parentheses around type
pub fn around_single_bound_ptr() -> *const (dyn Trait) {
    panic!()
}

pub fn around_multi_bound_dyn_fn_output() -> &'static dyn FnOnce() -> (impl Send + Sync) {
    &|| ()
}

//~v ERROR unnecessary parentheses around type
pub fn around_single_bound_dyn_fn_output() -> &'static dyn FnOnce() -> (impl Send) {
    &|| ()
}

pub fn around_dyn_fn_output_given_more_bounds() -> &'static (dyn FnOnce() -> (impl Send) + Sync) {
    &|| ()
}

pub fn around_multi_bound_impl_fn_output() -> impl FnOnce() -> (impl Send + Sync) {
    || ()
}

//~v ERROR unnecessary parentheses around type
pub fn around_single_bound_impl_fn_output() -> impl FnOnce() -> (impl Send) {
    || ()
}

pub fn around_impl_fn_output_given_more_bounds() -> impl FnOnce() -> (impl Send) + Sync {
    || ()
}

//~v ERROR unnecessary parentheses around type
pub fn around_dyn_bound() -> &'static dyn (FnOnce()) {
    &|| ()
}

//~v ERROR unnecessary parentheses around type
pub fn around_impl_trait_bound() -> impl (FnOnce()) {
    || ()
}

// these parens aren't strictly required but they help disambiguate => no lint
pub fn around_fn_bound_with_explicit_ret_ty() -> impl (Fn() -> ()) + Send {
    || ()
}

//~v ERROR unnecessary parentheses around type
pub fn around_fn_bound_with_implicit_ret_ty() -> impl (Fn()) + Send {
    || ()
}

//~v ERROR unnecessary parentheses around type
pub fn around_last_fn_bound_with_explicit_ret_ty() -> impl Send + (Fn() -> ()) {
    || ()
}

//~v ERROR unnecessary parentheses around type
pub fn around_regular_bound1() -> &'static (dyn (Send) + Sync) {
    &|| ()
}

//~v ERROR unnecessary parentheses around type
pub fn around_regular_bound2() -> &'static (dyn Send + (Sync)) {
    &|| ()
}

//~v ERROR unnecessary parentheses around type
pub fn around_regular_bound3() -> &'static (dyn Send + (::std::marker::Sync)) {
    &|| ()
}

pub fn parens_with_keyword(e: &[()]) -> i32 {
    if(true) {} //~ ERROR unnecessary parentheses around `if`
    while(true) {} //~ ERROR unnecessary parentheses around `while`
    for _ in(e) {} //~ ERROR unnecessary parentheses around `for`
    match(1) { _ => ()} //~ ERROR unnecessary parentheses around `match`
    return(1); //~ ERROR unnecessary parentheses around `return` value
}

macro_rules! baz {
    ($($foo:expr),+) => {
        ($($foo),*)
    };
}

macro_rules! unit {
    () => {
        ()
    };
}

struct One;

impl std::ops::Sub<One> for () {
    type Output = i32;
    fn sub(self, _: One) -> Self::Output {
        -1
    }
}

impl std::ops::Neg for One {
    type Output = i32;
    fn neg(self) -> Self::Output {
        -1
    }
}

pub const CONST_ITEM: usize = (10); //~ ERROR unnecessary parentheses around assigned value
pub static STATIC_ITEM: usize = (10); //~ ERROR unnecessary parentheses around assigned value

fn main() {
    foo();
    bar((true)); //~ ERROR unnecessary parentheses around function argument

    if (true) {} //~ ERROR unnecessary parentheses around `if` condition
    while (true) {} //~ ERROR unnecessary parentheses around `while` condition
    match (true) { //~ ERROR unnecessary parentheses around `match` scrutinee expression
        _ => {}
    }
    if let 1 = (1) {} //~ ERROR unnecessary parentheses around `let` scrutinee expression
    while let 1 = (2) {} //~ ERROR unnecessary parentheses around `let` scrutinee expression
    let v = X { y: false };
    // struct lits needs parens, so these shouldn't warn.
    if (v == X { y: true }) {}
    if (X { y: true } == v) {}
    if (X { y: false }.y) {}
    // this shouldn't warn, because the parens are necessary to disambiguate let chains
    if let true = (true && false) {}

    while (X { y: false }.foo(true)) {}
    while (true | X { y: false }.y) {}

    match (X { y: false }) {
        _ => {}
    }

    X { y: false }.foo((true)); //~ ERROR unnecessary parentheses around method argument

    let mut _a = (0); //~ ERROR unnecessary parentheses around assigned value
    _a = (0); //~ ERROR unnecessary parentheses around assigned value
    _a += (1); //~ ERROR unnecessary parentheses around assigned value

    let(mut _a) = 3; //~ ERROR unnecessary parentheses around pattern
    let (mut _a) = 3; //~ ERROR unnecessary parentheses around pattern
    let( mut _a) = 3; //~ ERROR unnecessary parentheses around pattern

    let(_a) = 3; //~ ERROR unnecessary parentheses around pattern
    let (_a) = 3; //~ ERROR unnecessary parentheses around pattern
    let( _a) = 3; //~ ERROR unnecessary parentheses around pattern

    let _a = baz!(3, 4);
    let _b = baz!(3);

    let _ = {
        (unit!() - One) //~ ERROR unnecessary parentheses around block return value
    } + {
        (unit![] - One) //~ ERROR unnecessary parentheses around block return value
    } + {
        // FIXME: false positive. This parenthesis is required.
        (unit! {} - One) //~ ERROR unnecessary parentheses around block return value
    };

    // Do *not* lint around `&raw` (but do lint when `&` creates a reference).
    let mut x = 0;
    let _r = (&x); //~ ERROR unnecessary parentheses
    let _r = (&mut x); //~ ERROR unnecessary parentheses
    let _r = (&raw const x);
    let _r = (&raw mut x);
}
