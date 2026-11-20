fn main() {
    let x: &Option<Box<i32>> = &Some(Box::new(0));

    match x {
    //~^ ERROR cannot move out of `x` as enum variant `Some` which is behind a shared reference
        &Some(_y) => (),
        &None => (),
    }
}
