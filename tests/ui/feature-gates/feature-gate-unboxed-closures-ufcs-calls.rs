#![allow(dead_code)]

fn foo<F: Fn()>(mut f: F) {
    Fn::call(&f, ()); //~ ERROR use of unstable library feature `fn_traits`
    FnMut::call_mut(&mut f, ()); //~ ERROR use of unstable library feature `fn_traits`
    FnOnce::call_once(f, ()); //~ ERROR use of unstable library feature `fn_traits`
}

fn main() {}
