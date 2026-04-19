// Regression test for #154878
//@ revisions: cpass1 cpass2

pub fn main() {
    let x = 42.0;
    #[expect(invalid_nan_comparisons)]
    let _b = x == f32::NAN;
}
