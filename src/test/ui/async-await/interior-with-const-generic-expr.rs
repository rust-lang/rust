// edition:2018
// run-pass

#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![allow(unused)]

fn main() {
    let x = test();
}

fn concat<const A: usize, const B: usize>(a: [f32; A], b: [f32; B]) -> [f32; A + B] {
    todo!()
}

async fn reverse<const A: usize>(x: [f32; A]) -> [f32; A] {
    todo!()
}

async fn test() {
    let a = [0.0];
    let b = [1.0, 2.0];
    let ab = concat(a,b);
    let ba = reverse(ab).await;
    println!("{:?}", ba);
}
