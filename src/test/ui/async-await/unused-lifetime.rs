// edition:2018

// Avoid spurious warnings of unused lifetime. The below async functions
// are desugered to have an unused lifetime
// but we don't want to warn about that as there's nothing they can do about it.

#![deny(unused_lifetimes)]
#![allow(dead_code)]

pub async fn october(s: &str) {
    println!("{}", s);
}

pub async fn async_fn(&mut ref s: &mut[i32]) {
    println!("{:?}", s);
}

macro_rules! foo_macro {
    () => {
        pub async fn async_fn_in_macro(&mut ref _s: &mut[i32]) {}
    };
}

foo_macro!();

pub async fn func_with_unused_lifetime<'a>(s: &'a str) {
    //~^ ERROR lifetime parameter `'a` never used
    println!("{}", s);
}

pub async fn func_with_two_unused_lifetime<'a, 'b>(s: &'a str, t: &'b str) {
    //~^ ERROR lifetime parameter `'a` never used
    //~^^ ERROR lifetime parameter `'b` never used
    println!("{}", s);
}

pub async fn func_with_unused_lifetime_in_two_params<'c>(s: &'c str, t: &'c str) {
    //~^ ERROR lifetime parameter `'c` never used
    println!("{}", s);
}

fn main() {}
