#![warn(clippy::suspicious_map)]

fn main() {
    let _ = (0..3).map(|x| x + 2).count();
}
