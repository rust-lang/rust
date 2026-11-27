//@ only-x86_64
//@ run-pass

#[target_feature(enable = "sse2")]
fn foo() -> bool {
    true
}

#[target_feature(enable = "sse2")]
fn bar() -> fn() -> bool {
    foo
}

fn main() {
    if !std::is_x86_feature_detected!("sse2") {
        return;
    }
    let f = unsafe { bar() };
    assert!(f());
}
