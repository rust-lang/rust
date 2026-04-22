// Regression test for #154878
//@ revisions: bpass1 bpass2

pub fn main() {
    let x = 42.0;
    #[expect(invalid_nan_comparisons)]
    let _b = x == f32::NAN;
}
