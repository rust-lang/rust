#![warn(clippy::maybe_misused_cfg)]

fn main() {
    #[cfg(features = "not-really-a-feature")]
    let _ = 1 + 2;

    #[cfg(all(feature = "right", features = "wrong"))]
    let _ = 1 + 2;

    #[cfg(all(features = "wrong1", any(feature = "right", features = "wrong2", feature, features)))]
    let _ = 1 + 2;
}
