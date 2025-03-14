use test::{Bencher, black_box};

const CHARS: [char; 9] = ['0', 'x', '2', '5', 'A', 'f', '7', '8', '9'];
const RADIX: [u32; 5] = [2, 8, 10, 16, 32];

#[bench]
fn bench_to_digit_radix_2(b: &mut Bencher) {
    b.iter(|| CHARS.iter().cycle().take(10_000).map(|c| black_box(c).to_digit(2)).min())
}

#[bench]
fn bench_to_digit_radix_10(b: &mut Bencher) {
    b.iter(|| CHARS.iter().cycle().take(10_000).map(|c| black_box(c).to_digit(10)).min())
}

#[bench]
fn bench_to_digit_radix_16(b: &mut Bencher) {
    b.iter(|| CHARS.iter().cycle().take(10_000).map(|c| black_box(c).to_digit(16)).min())
}

#[bench]
fn bench_to_digit_radix_36(b: &mut Bencher) {
    b.iter(|| CHARS.iter().cycle().take(10_000).map(|c| black_box(c).to_digit(36)).min())
}

#[bench]
fn bench_to_digit_radix_var(b: &mut Bencher) {
    b.iter(|| {
        CHARS
            .iter()
            .cycle()
            .zip(RADIX.iter().cycle())
            .take(10_000)
            .map(|(c, radix)| black_box(c).to_digit(*radix))
            .min()
    })
}

#[bench]
fn bench_to_ascii_uppercase(b: &mut Bencher) {
    b.iter(|| CHARS.iter().cycle().take(10_000).map(|c| black_box(c).to_ascii_uppercase()).min())
}

#[bench]
fn bench_to_ascii_lowercase(b: &mut Bencher) {
    b.iter(|| CHARS.iter().cycle().take(10_000).map(|c| black_box(c).to_ascii_lowercase()).min())
}

#[bench]
fn bench_ascii_mix_to_uppercase(b: &mut Bencher) {
    b.iter(|| {
        (0..=255).cycle().take(10_000).map(|b| black_box(char::from(b)).to_uppercase()).count()
    })
}

#[bench]
fn bench_ascii_mix_to_lowercase(b: &mut Bencher) {
    b.iter(|| {
        (0..=255).cycle().take(10_000).map(|b| black_box(char::from(b)).to_lowercase()).count()
    })
}

#[bench]
fn bench_ascii_char_to_uppercase(b: &mut Bencher) {
    b.iter(|| {
        (0..=127).cycle().take(10_000).map(|b| black_box(char::from(b)).to_uppercase()).count()
    })
}

#[bench]
fn bench_ascii_char_to_lowercase(b: &mut Bencher) {
    b.iter(|| {
        (0..=127).cycle().take(10_000).map(|b| black_box(char::from(b)).to_lowercase()).count()
    })
}

#[bench]
fn bench_non_ascii_char_to_uppercase(b: &mut Bencher) {
    b.iter(|| {
        (128..=255).cycle().take(10_000).map(|b| black_box(char::from(b)).to_uppercase()).count()
    })
}

#[bench]
fn bench_non_ascii_char_to_lowercase(b: &mut Bencher) {
    b.iter(|| {
        (128..=255).cycle().take(10_000).map(|b| black_box(char::from(b)).to_lowercase()).count()
    })
}
