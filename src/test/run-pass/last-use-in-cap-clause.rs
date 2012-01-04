// Make sure #1399 stays fixed

fn foo() -> lambda() -> int {
    let k = ~22;
    let _u = {a: k};
    ret lambda[move k]() -> int { 22 };
}

fn main() {
    assert foo()() == 22;
}
