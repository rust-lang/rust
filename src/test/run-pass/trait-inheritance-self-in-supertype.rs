// Test for issue #4183: use of Self in supertraits.

pub static FUZZY_EPSILON: float = 0.1;

pub trait FuzzyEq<Eps> {
    fn fuzzy_eq(&self, other: &Self) -> bool;
    fn fuzzy_eq_eps(&self, other: &Self, epsilon: &Eps) -> bool;
}

trait Float: FuzzyEq<Self> {
    fn two_pi() -> Self;
}

impl FuzzyEq<f32> for f32 {
    fn fuzzy_eq(&self, other: &f32) -> bool {
        self.fuzzy_eq_eps(other, &(FUZZY_EPSILON as f32))
    }

    fn fuzzy_eq_eps(&self, other: &f32, epsilon: &f32) -> bool {
        f32::abs(*self - *other) < *epsilon
    }
}

impl Float for f32 {
    fn two_pi() -> f32 { 6.28318530717958647692528676655900576_f32 }
}

impl FuzzyEq<f64> for f64 {
    fn fuzzy_eq(&self, other: &f64) -> bool {
        self.fuzzy_eq_eps(other, &(FUZZY_EPSILON as f64))
    }

    fn fuzzy_eq_eps(&self, other: &f64, epsilon: &f64) -> bool {
        f64::abs(*self - *other) < *epsilon
    }
}

impl Float for f64 {
    fn two_pi() -> f64 { 6.28318530717958647692528676655900576_f64 }
}

fn compare<F:Float>(f1: F) -> bool {
    let f2 = Float::two_pi();
    f1.fuzzy_eq(&f2)
}

pub fn main() {
    assert!(compare::<f32>(6.28318530717958647692528676655900576));
    assert!(compare::<f32>(6.29));
    assert!(compare::<f32>(6.3));
    assert!(compare::<f32>(6.19));
    assert!(!compare::<f32>(7.28318530717958647692528676655900576));
    assert!(!compare::<f32>(6.18));

    assert!(compare::<f64>(6.28318530717958647692528676655900576));
    assert!(compare::<f64>(6.29));
    assert!(compare::<f64>(6.3));
    assert!(compare::<f64>(6.19));
    assert!(!compare::<f64>(7.28318530717958647692528676655900576));
    assert!(!compare::<f64>(6.18));
}