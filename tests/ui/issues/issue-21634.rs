//@ run-pass
#![allow(stable_features)]

#![feature(cfg_target_feature)]

#[cfg(any(not(target_arch = "x86"), target_feature = "sse2"))]
fn main() {
    if let Ok(x) = "3.1415".parse::<f64>() {
        assert_eq!(false, x <= 0.0);
    }
    if let Ok(x) = "3.1415".parse::<f64>() {
        assert_eq!(3.1415, x + 0.0);
    }
    if let Ok(mut x) = "3.1415".parse::<f64>() {
        assert_eq!(8.1415, { x += 5.0; x });
    }
}

#[cfg(all(target_arch = "x86", not(target_feature = "sse2")))]
fn main() {}
