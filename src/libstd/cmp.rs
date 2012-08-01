/// Additional general-purpose comparison functionality.

const fuzzy_epsilon: float = 1.0e-6;

trait fuzzy_eq {
    pure fn fuzzy_eq(&&other: self) -> bool;
}

impl fuzzy_eq of fuzzy_eq for float {
    pure fn fuzzy_eq(&&other: float) -> bool {
        ret float::abs(self - other) < fuzzy_epsilon;
    }
}

impl fuzzy_eq of fuzzy_eq for f32 {
    pure fn fuzzy_eq(&&other: f32) -> bool {
        ret f32::abs(self - other) < (fuzzy_epsilon as f32);
    }
}

impl fuzzy_eq of fuzzy_eq for f64 {
    pure fn fuzzy_eq(&&other: f64) -> bool {
        ret f64::abs(self - other) < (fuzzy_epsilon as f64);
    }
}

#[test]
fn test_fuzzy_equals() {
    assert ((1.0).fuzzy_eq(1.0));
    assert ((1.0f32).fuzzy_eq(1.0f32));
    assert ((1.0f64).fuzzy_eq(1.0f64));
}

