// run-pass
// pretty-expanded FIXME #23616

pub fn main() {
    let x = &Some(1);
    match x {
        &Some(_) => (),
        &None => (),
    }
}
