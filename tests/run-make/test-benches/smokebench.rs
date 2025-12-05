#![feature(test)]
extern crate test;

#[bench]
fn smoke_yesiter(b: &mut test::Bencher) {
    let mut i = 0usize;
    b.iter(|| {
        i = i.wrapping_add(1);
        i
    })
}

#[bench]
fn smoke_noiter(_: &mut test::Bencher) {}
