// Issue #1763 - infer types correctly
// error-pattern:explicit failure

type actor<T> = {
    unused: bool
};

fn act2<T>() -> actor<T> {
    fail;
}

fn main() {
    let a: actor<int> = act2();
}
