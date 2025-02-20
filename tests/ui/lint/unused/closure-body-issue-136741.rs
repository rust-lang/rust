//@ run-rustfix
#![deny(unused_parens)]
#![deny(unused_braces)]
pub fn main() {
    let _closure = |x: i32, y: i32| { x * (x + (y * 2)) }; //~ ERROR unnecessary braces around closure body
    let _ = || (0 == 0); //~ ERROR unnecessary parentheses around closure body
    let _ = (0..).find(|n| (n % 2 == 0)); //~ ERROR unnecessary parentheses around closure body
    let _ = (0..).find(|n| {n % 2 == 0}); //~ ERROR unnecessary braces around closure body
    let _ = || {
        _ = 0;
        (0 == 0) //~ ERROR unnecessary parentheses around block return value
    };
}
