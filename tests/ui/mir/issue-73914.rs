//@ build-pass
//@ compile-flags:-Copt-level=0
//@ edition:2018

struct S<T>(std::marker::PhantomData<T>);

impl<T> std::ops::Deref for S<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        todo!()
    }
}
impl<T> std::ops::DerefMut for S<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        todo!()
    }
}

async fn new() -> S<u64> {
    todo!()
}

async fn crash() {
    *new().await = 1 + 1;
}

fn main() {
    let _ = crash();
}
