//@ run-pass
//@ compile-flags:-Zpolymorphize=on

fn fop<T>() {}

fn bar<T>() -> &'static fn() {
    &(fop::<T> as fn())
}
pub const FN: &'static fn() = &(fop::<i32> as fn());

fn main() {
    bar::<u32>();
    bar::<i32>();
    (FN)();
}
