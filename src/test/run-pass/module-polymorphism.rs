// This isn't really xfailed; it's used by the
// module-polymorphism.rc test
// xfail-test

fn main() {
    // All of these functions are defined by a single module
    // source file but instantiated for different types
    assert my_float::template::plus(1.0f, 2.0f) == 3.0f;
    assert my_f64::template::plus(1.0f64, 2.0f64) == 3.0f64;
    assert my_f32::template::plus(1.0f32, 2.0f32) == 3.0f32;
}