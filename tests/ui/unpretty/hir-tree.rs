//@ build-pass
//@ compile-flags: -o - -Zunpretty=hir-tree
//@ check-stdout
//@ dont-check-compiler-stdout
//@ dont-check-compiler-stderr

fn main() {
    println!("Hello, Rustaceans!");
}

//~? RAW Hello, Rustaceans!
