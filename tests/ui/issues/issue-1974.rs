// run-pass
// Issue 1974
// Don't double free the condition allocation
// pretty-expanded FIXME #23616

pub fn main() {
    let s = "hej".to_string();
    while s != "".to_string() {
        return;
    }
}
