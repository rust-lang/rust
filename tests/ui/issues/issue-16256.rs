//@ run-pass

fn main() {
    let mut buf = Vec::new();
    |c: u8| buf.push(c); //~ WARN unused closure that must be used
}
