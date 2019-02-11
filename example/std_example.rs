use std::io::Write;

fn main() {
    let _ = ::std::iter::repeat('a' as u8).take(10).collect::<Vec<_>>();
    let stderr = ::std::io::stderr();
    let mut stderr = stderr.lock();

    writeln!(stderr, "some {} text", "<unknown>").unwrap();
}
