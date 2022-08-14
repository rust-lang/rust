// compile-flags: -Zunpretty=hir
// check-pass

#![feature(let_else)]

fn foo(x: Option<u32>) {
    let Some(_) = x else { panic!() };
}

fn main() {}
