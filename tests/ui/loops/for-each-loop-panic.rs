//@ run-fail
//@ error-pattern:moop
//@ ignore-emscripten no processes

fn main() {
    for _ in 0_usize..10_usize {
        panic!("moop");
    }
}
