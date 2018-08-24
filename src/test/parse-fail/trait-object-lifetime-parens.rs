// compile-flags: -Z parse-only -Z continue-parse-after-error

fn f<T: Copy + ('a)>() {} //~ ERROR parenthesized lifetime bounds are not supported

fn main() {
    let _: Box<Copy + ('a)>; //~ ERROR parenthesized lifetime bounds are not supported
    let _: Box<('a) + Copy>; //~ ERROR expected type, found `'a`
}
