fn main() {
    let range = [0i32].as_ptr_range();
    for _ in range {}
    //~^ ERROR: the trait bound `*const i32: Step` is not satisfied
}
