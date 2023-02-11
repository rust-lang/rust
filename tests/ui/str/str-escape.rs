// check-pass
fn main() {
    let s = "\

             ";
    //~^^^ WARNING multiple lines skipped by escaped newline
    let s = "foo\
             bar
             ";
    //~^^^ WARNING non-ASCII whitespace symbol '\u{a0}' is not skipped
}
