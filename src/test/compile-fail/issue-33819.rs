fn main() {
    let mut op = Some(2);
    match op {
        Some(ref v) => { let a = &mut v; },
        //~^ ERROR:cannot borrow immutable
        //~| use `ref mut v` here to make mutable
        None => {},
    }
}
