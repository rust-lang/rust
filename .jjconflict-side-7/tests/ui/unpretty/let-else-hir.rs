//@ compile-flags: -Zunpretty=hir
//@ check-pass
//@ edition: 2015



fn foo(x: Option<u32>) {
    let Some(_) = x else { panic!() };
}

fn main() {}
