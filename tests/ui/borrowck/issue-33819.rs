//@ check-pass
fn main() {
    let mut op = Some(2);
    match op {
        Some(ref v) => { let a = &mut v; },
        //~^ WARNING cannot borrow `v` as mutable, as it is not declared as mutable
        //~| HELP try removing `&mut` here
        None => {},
    }
}
