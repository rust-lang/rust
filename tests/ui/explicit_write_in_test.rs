//@ check-pass
#![warn(clippy::explicit_write)]

#[test]
fn test() {
    use std::io::Write;
    writeln!(std::io::stderr(), "I am an explicit write.").unwrap();
    eprintln!("I am not an explicit write.");
}
