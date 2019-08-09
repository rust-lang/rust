fn main() {
    let mut op = Some(2);
    match op {
        Some(ref v) => { let a = &mut v; },
        //~^ ERROR cannot borrow `v` as mutable, as it is not declared as mutable
        None => {},
    }
}
