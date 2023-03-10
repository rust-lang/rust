// check-pass
// compile-flags: -Ztrait-solver=next

// HIR typeck ends up equating `<_#0i as Add>::Output == _#0i`.
// Want to make sure that we emit an alias-eq goal for this,
// instead of treating it as a type error and bailing.

fn test() {
    // fallback
    let x = 1 + 2;
}

fn test2() -> u32 {
    // expectation from return ty
    1 + 2
}

fn main() {}
