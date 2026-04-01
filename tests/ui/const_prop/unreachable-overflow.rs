// Use `build-pass` to ensure const-prop lint runs.
//@ build-pass

fn main() {
    let x = 2u32;
    let y = 3u32;
    if y <= x {
        dbg!(x - y);
    }
}
