// Regression test for https://github.com/rust-lang/rust/issues/121508.
struct Struct<T>(T);

impl<T> std::ops::Deref for Struct<T> {
    type Target = dyn Fn(T);
    fn deref(&self) -> &assert_mem_uninitialized_valid::Target {
        //~^ERROR: use of unresolved module or unlinked crate
        unimplemented!()
    }
}

fn main() {
    let f = Struct(Default::default());
    f(0);
    f(0);
}
