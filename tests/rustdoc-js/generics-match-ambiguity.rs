pub struct Wrap<T, U = ()>(pub T, pub U);

pub fn foo(a: Wrap<i32>, b: Wrap<i32, u32>) {}
pub fn bar(a: Wrap<i32, u32>, b: Wrap<i32>) {}

pub struct W2<T>(pub T);
pub struct W3<T, U = ()>(pub T, pub U);

pub fn baaa(a: W3<i32>, b: W3<i32, u32>) {}
pub fn baab(a: W3<i32, u32>, b: W3<i32>) {}
pub fn baac(a: W2<W3<i32>>, b: W3<i32, u32>) {}
pub fn baad(a: W2<W3<i32, u32>>, b: W3<i32>) {}
pub fn baae(a: W3<i32>, b: W2<W3<i32, u32>>) {}
pub fn baaf(a: W3<i32, u32>, b: W2<W3<i32>>) {}
pub fn baag(a: W2<W3<i32>>, b: W2<W3<i32, u32>>) {}
pub fn baah(a: W2<W3<i32, u32>>, b: W2<W3<i32>>) {}
//
