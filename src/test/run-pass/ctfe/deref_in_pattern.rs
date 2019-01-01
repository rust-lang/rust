// run-pass

const A: [u8; 4] = *b"fooo";

fn main() {
    match *b"xxxx" {
        A => {},
        _ => {}
    }
}
