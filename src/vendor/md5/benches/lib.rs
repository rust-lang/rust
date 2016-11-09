#![feature(test)]

extern crate md5;
extern crate test;

#[bench] fn compute_0001000(bencher: &mut test::Bencher) { compute(    1000, bencher); }
#[bench] fn compute_0010000(bencher: &mut test::Bencher) { compute(   10000, bencher); }
#[bench] fn compute_0100000(bencher: &mut test::Bencher) { compute(  100000, bencher); }
#[bench] fn compute_1000000(bencher: &mut test::Bencher) { compute(1_000000, bencher); }

fn compute(size: usize, bencher: &mut test::Bencher) {
    let data = &vec![0xFFu8; size][..];
    bencher.iter(|| {
        test::black_box(md5::compute(data));
    });
}
