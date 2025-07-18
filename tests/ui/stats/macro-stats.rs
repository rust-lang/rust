//@ check-pass
//@ compile-flags: -Zmacro-stats

#[test]
fn test_foo() {
    let what = "this";
    let how = "completely";
    let when = "immediately";
    println!("{what} disappears {how} and {when}");
}

#[rustfmt::skip] // non-macro attr, ignored by `-Zmacro-stats`
fn rustfmt_skip() {
    // Nothing to see here.
}

#[derive(Default, Clone, Copy, Hash)]
enum E1 {
    #[default] // non-macro attr, ignored by `-Zmacro-stats`
    A,
    B,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct E2 {
    a: u32,
    b: String,
    c: Vec<bool>,
}

#[derive(Clone)] struct S0;
#[derive(Clone)] struct S1(u32);
#[derive(Clone)] struct S2(u32, u32);
#[derive(Clone)] struct S3(u32, u32, u32);
#[derive(Clone)] struct S4(u32, u32, u32, u32);
#[derive(Clone)] struct S5(u32, u32, u32, u32, u32);

macro_rules! u32 {
    () => { u32 }
}

macro_rules! none {
    () => { None }
}
fn opt(x: Option<u32>) {
    match x {
        Some(_) => {}
        none!() => {}           // AstFragmentKind::Pat
    }
}

macro_rules! long_name_that_fits_on_a_single_line {
    () => {}
}
long_name_that_fits_on_a_single_line!();

macro_rules! long_name_that_doesnt_fit_on_one_line {
    ($t:ty) => {
        fn f(_: $t) {}
    }
}
long_name_that_doesnt_fit_on_one_line!(u32!()); // AstFragmentKind::{Items,Ty}

macro_rules! trait_tys {
    () => {
        type A;
        type B;
    }
}
trait Tr {
    trait_tys!();               // AstFragmentKind::TraitItems
}

macro_rules! impl_const { () => { const X: u32 = 0; } }
struct U;
impl U {
    impl_const!();              // AstFragmentKind::ImplItems
}

macro_rules! trait_impl_tys {
    () => {
        type A = u32;
        type B = bool;
    }
}
struct Tr1;
impl Tr for Tr1 {
    trait_impl_tys!();          // AstFragment::TraitImplItems
}

macro_rules! foreign_item {
    () => { fn fc(a: u32) -> u32; }
}
extern "C" {
    foreign_item!();            // AstFragment::ForeignItems
}

// Include macros are ignored by `-Zmacro-stats`.
mod includes {
    mod z1 {
        include!("auxiliary/include.rs");
    }
    mod z2 {
        std::include!("auxiliary/include.rs");
    }

    const B1: &[u8] = include_bytes!("auxiliary/include.rs");
    const B2: &[u8] = std::include_bytes!("auxiliary/include.rs");

    const S1: &str = include_str!("auxiliary/include.rs");
    const S2: &str = std::include_str!("auxiliary/include.rs");
}

fn main() {
    macro_rules! n99 {
        () => { 99 }
    }
    let x = n99!() + n99!();    // AstFragmentKind::Expr

    macro_rules! p {
        () => {
            // blah
            let x = 1;
            let y = x;
            let _ = y;
        }
    }
    p!();                       // AstFragmentKind::Stmts

    macro_rules! q {
        () => {};
        ($($x:ident),*) => { $( let $x: u32 = 12345; )* };
    }
    q!(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z);
}
