//@ build-pass
//@ compile-flags: -o - -Zunpretty=hir-tree
//@ check-stdout
//@ dont-check-compiler-stdout
//@ dont-check-compiler-stderr
//@ regex-error-pattern: Hello, Rustaceans!

fn main() {
    println!("Hello, Rustaceans!");
}
