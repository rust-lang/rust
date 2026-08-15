//@ run-pass

const I: std::iter::Empty<u32> = std::iter::empty();

fn main() {
    for i in I {
        panic!("magical value creation: {}", i);
    }
}
