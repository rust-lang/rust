use core::any::*;
use test::{Bencher, black_box};

#[bench]
fn bench_downcast_ref(b: &mut Bencher) {
    b.iter(|| {
        let mut x = 0;
        let mut y = &mut x as &mut dyn Any;
        black_box(&mut y);
        black_box(y.downcast_ref::<isize>() == Some(&0));
    });
}
