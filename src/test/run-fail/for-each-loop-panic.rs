// error-pattern:moop

fn main() {
    for _ in 0_usize..10_usize {
        panic!("moop");
    }
}
