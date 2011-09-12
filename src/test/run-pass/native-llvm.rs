// xfail-test

native "llvm" mod llvm {
    fn thesqrt(n: float) -> float = "sqrt.f64";
}

fn main() {
    let s = llvm::thesqrt(4.0);
    assert 1.9 < s && s < 2.1;
}