// run-pass
// compile-flags: -Zpolymorphize=on -Zmir-opt-level=3

fn caller<T, U>() -> &'static usize {
    callee::<U>()
}

fn callee<T>() -> &'static usize {
    &std::mem::size_of::<T>()
}

fn main() {
    assert_eq!(caller::<(), ()>(), &0);
}
