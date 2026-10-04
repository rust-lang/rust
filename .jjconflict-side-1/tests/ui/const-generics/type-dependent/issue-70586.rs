//@ check-pass
use std::marker::PhantomData;

// This namespace is necessary for the ICE to trigger
struct Namespace;

impl Namespace {
    pub fn const_chunks_exact<T, const N: usize>() -> ConstChunksExact<'static, T, N> {
        ConstChunksExact { inner: PhantomData }
    }
}


#[derive(Debug)]
pub struct ConstChunksExact<'a, T, const N: usize> {
    inner:  PhantomData<&'a T>
}

impl <'a, T, const N: usize> Iterator for ConstChunksExact<'a, T, { N }> {
    type Item = &'a [T; N];

    fn next(&mut self) -> Option<Self::Item> {
        unreachable!()
    }
}

fn main() {
    let mut chunks = Namespace::const_chunks_exact::<i32, 3usize>();
    let _next: &[i32; 3] = chunks.next().unwrap();
}
