use test::{Bencher, black_box};

use super::corpora;

#[bench]
fn chars_advance_by_1000(b: &mut Bencher) {
    b.iter(|| black_box(corpora::ru::LARGE).chars().advance_by(1000));
}

#[bench]
fn chars_advance_by_0010(b: &mut Bencher) {
    b.iter(|| black_box(corpora::ru::LARGE).chars().advance_by(10));
}

#[bench]
fn chars_advance_by_0001(b: &mut Bencher) {
    b.iter(|| black_box(corpora::ru::LARGE).chars().advance_by(1));
}

mod chars_sum {
    use super::*;

    fn bench(b: &mut Bencher, corpus: &str) {
        b.iter(|| corpus.chars().map(|c| c as u32).sum::<u32>())
    }

    #[bench]
    fn en(b: &mut Bencher) {
        bench(b, corpora::en::HUGE);
    }

    #[bench]
    fn zh(b: &mut Bencher) {
        bench(b, corpora::zh::HUGE);
    }

    #[bench]
    fn ru(b: &mut Bencher) {
        bench(b, corpora::zh::HUGE);
    }

    #[bench]
    fn emoji(b: &mut Bencher) {
        bench(b, corpora::zh::HUGE);
    }
}

mod chars_sum_rev {
    use super::*;

    fn bench(b: &mut Bencher, corpus: &str) {
        b.iter(|| corpus.chars().rev().map(|c| c as u32).sum::<u32>())
    }

    #[bench]
    fn en(b: &mut Bencher) {
        bench(b, corpora::en::HUGE);
    }

    #[bench]
    fn zh(b: &mut Bencher) {
        bench(b, corpora::zh::HUGE);
    }

    #[bench]
    fn ru(b: &mut Bencher) {
        bench(b, corpora::zh::HUGE);
    }

    #[bench]
    fn emoji(b: &mut Bencher) {
        bench(b, corpora::zh::HUGE);
    }
}
