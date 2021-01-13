// edition:2018
// check-pass
// revisions: full min
#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]

async fn foo<const N: usize>(arg: [u8; N]) -> usize { arg.len() }

async fn bar<const N: usize>() -> [u8; N] {
    [0; N]
}

trait Trait<const N: usize> {
    fn fynn(&self) -> usize;
}
impl<const N: usize> Trait<N> for [u8; N] {
    fn fynn(&self) -> usize {
        N
    }
}
async fn baz<const N: usize>() -> impl Trait<N> {
    [0; N]
}

async fn biz<const N: usize>(v: impl Trait<N>) -> usize {
    v.fynn()
}

async fn user<const N: usize>() {
    let _ = foo::<N>(bar().await).await;
    let _ = biz(baz::<N>().await).await;
}

fn main() { }
