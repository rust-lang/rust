// compile-flags: -Zpolymorphize=on -Zmir-opt-level=4

fn caller<T, U>() -> &'static usize {
    callee::<U>()
}

fn callee<T>() -> &'static usize {
    &std::mem::size_of::<T>()
    //~^ ERROR: cannot return reference to temporary value
}

fn main() {
    assert_eq!(caller::<(), ()>(), &0);
}
