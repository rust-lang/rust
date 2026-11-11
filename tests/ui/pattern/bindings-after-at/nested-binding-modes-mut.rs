fn main() {
    let mut is_mut @ not_mut = 42;
    &mut is_mut;
    &mut not_mut;
    //~^ ERROR cannot borrow

    let not_mut @ mut is_mut = 42;
    &mut is_mut;
    &mut not_mut;
    //~^ ERROR cannot borrow
}
