// xfail-test #2443
// exec-env:RUST_POISON_ON_FREE

fn it_takes_two(x: @int, -y: @int) -> int {
    free(y);
    #debug["about to deref"];
    *x
}

fn free<T>(-_t: T) {
}

fn main() {
    let z = @3;
    assert 3 == it_takes_two(z, z);
}
