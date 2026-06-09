//@ run-pass
#![deny(meta_variable_misuse)]

macro_rules! foo {
    ($($m:ident $($f:ident $v:tt)+),*) => {
        $($(macro_rules! $f { () => { $v } })+)*
        $(macro_rules! $m { () => { $(fn $f() -> i32 { $v })+ } })*
    }
}

foo!(m a 1 b 2, n c 3);
m!();
n!();

macro_rules! no_shadow {
    ($x:tt) => { macro_rules! bar { ($x:tt) => { 42 }; } };
}
no_shadow!(z);

macro_rules! make_plus {
    ($n: ident $x:expr) => { macro_rules! $n { ($y:expr) => { $x + $y }; } };
}
make_plus!(add3 3);

fn main() {
    assert_eq!(a!(), 1);
    assert_eq!(b!(), 2);
    assert_eq!(c!(), 3);
    assert_eq!(a(), 1);
    assert_eq!(b(), 2);
    assert_eq!(c(), 3);
    assert_eq!(bar!(z:tt), 42);
    assert_eq!(add3!(9), 12);
}
