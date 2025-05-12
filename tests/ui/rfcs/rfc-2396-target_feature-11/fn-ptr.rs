//@ only-x86_64

#[target_feature(enable = "avx")]
fn foo_avx() {}

#[target_feature(enable = "sse2")]
fn foo() {}

#[target_feature(enable = "sse2")]
fn bar() {
    let foo: fn() = foo; // this is OK, as we have the necessary target features.
    let foo: fn() = foo_avx; //~ ERROR mismatched types
}

fn main() {
    if std::is_x86_feature_detected!("sse2") {
        unsafe {
            bar();
        }
    }
    let foo: fn() = foo; //~ ERROR mismatched types
}
