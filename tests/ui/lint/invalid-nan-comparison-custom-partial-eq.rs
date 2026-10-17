//@ check-pass

struct NotFloat;

impl PartialEq<f64> for NotFloat {
    fn eq(&self, _: &f64) -> bool {
        true
    }
}

impl PartialEq<NotFloat> for f64 {
    fn eq(&self, _: &NotFloat) -> bool {
        true
    }
}

fn main() {
    let value = NotFloat;
    let _ = value == f64::NAN;
    //~^ WARN incorrect NaN comparison
    let _ = f64::NAN != value;
    //~^ WARN incorrect NaN comparison
    let _ = &value != &f64::NAN;
    //~^ WARN incorrect NaN comparison
}
