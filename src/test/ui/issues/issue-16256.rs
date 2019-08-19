// run-pass
// pretty-expanded FIXME #23616

fn main() {
    let mut buf = Vec::new();
    |c: u8| buf.push(c);
}
