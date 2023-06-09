// compile-flags: -Zunpretty=hir
// check-pass



fn foo(x: Option<u32>) {
    let Some(_) = x else { panic!() };
}

fn main() {}
