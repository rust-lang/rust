// check-pass
// compile-flags: -Zunpretty=hir

// https://github.com/rust-lang/rust/issues/87577

#[derive(Debug)]
struct S<#[cfg(feature = "alloc")] N: A<T>>;

fn main() {
    let s = S;
}
