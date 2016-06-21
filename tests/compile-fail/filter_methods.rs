#![feature(plugin)]
#![plugin(clippy)]

#![deny(clippy, clippy_pedantic)]
fn main() {
    let _: Vec<_> = vec![5; 6].into_iter() //~ERROR called `filter(p).map(q)` on an `Iterator`
                              .filter(|&x| x == 0)
                              .map(|x| x * 2)
                              .collect();

    let _: Vec<_> = vec![5i8; 6].into_iter() //~ERROR called `filter(p).flat_map(q)` on an `Iterator`
                                .filter(|&x| x == 0)
                                .flat_map(|x| x.checked_mul(2))
                                .collect();

    let _: Vec<_> = vec![5i8; 6].into_iter() //~ERROR called `filter_map(p).flat_map(q)` on an `Iterator`
                                .filter_map(|x| x.checked_mul(2))
                                .flat_map(|x| x.checked_mul(2))
                                .collect();
}
