fn main() {
    let mut op = Some(2);
    match op {
        Some(ref v) => { let a = &mut v; },
        //~^ ERROR:cannot borrow immutable
        //~| cannot borrow mutably
        None => {},
    }
}
