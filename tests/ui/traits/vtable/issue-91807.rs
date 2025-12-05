//@ check-pass
//@ incremental

struct Struct<T>(T);

impl<T> std::ops::Deref for Struct<T> {
    type Target = dyn Fn(T);
    fn deref(&self) -> &Self::Target {
        unimplemented!()
    }
}

fn main() {
    let f = Struct(Default::default());
    f(0);
    f(0);
}
