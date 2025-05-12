//@ run-pass

pub fn main() {
    let x = &Some(1);
    match x {
        &Some(_) => (),
        &None => (),
    }
}
