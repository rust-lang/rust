//@ run-pass

const fn make_nans() -> (f64, f64, f32, f32) {
    let nan1 = f64::from_bits(0x7FF0_0001_0000_0001);
    let nan2 = f64::from_bits(0x7FF0_0000_0000_0001);

    let nan1_32 = nan1 as f32;
    let nan2_32 = nan2 as f32;

    (nan1, nan2, nan1_32, nan2_32)
}

static NANS: (f64, f64, f32, f32) = make_nans();

fn main() {
    let (nan1, nan2, nan1_32, nan2_32) = NANS;

    assert!(nan1.is_nan());
    assert!(nan2.is_nan());

    assert!(nan1_32.is_nan());
    assert!(nan2_32.is_nan());
}
