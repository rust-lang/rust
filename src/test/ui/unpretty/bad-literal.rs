// compile-flags: -Zunpretty=hir
// check-fail

// In #100948 this caused an ICE with -Zunpretty=hir.
fn main() {
    1u;
    //~^ ERROR invalid suffix `u` for number literal
}
