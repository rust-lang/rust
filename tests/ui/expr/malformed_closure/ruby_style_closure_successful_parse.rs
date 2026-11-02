const x: usize =42;
fn main() {
    let p = Some(45).and_then({|x| //~ ERROR expected `FnOnce({integer})` signature, found `Option<usize>`
        1 + 1;
        Some(x * 2)
    });
}
