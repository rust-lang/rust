fn main() {
    let mut test = Vec::new();
    let rofl: &Vec<Vec<i32>> = &mut test;
    //~^ HELP consider changing this to be a mutable reference
    rofl.push(Vec::new());
    //~^ ERROR cannot borrow `*rofl` as mutable, as it is behind a `&` reference
    //~| NOTE `rofl` is a `&` reference, so the data it refers to cannot be borrowed as mutable
}
