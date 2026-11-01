// Use `build-pass` to ensure const-prop lint runs.
//@ build-pass

fn main() {
    [()][if false { 1 } else { return }]
}
