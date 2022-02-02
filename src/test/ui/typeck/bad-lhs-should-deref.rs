fn mut_ref() -> &'static mut Box<usize> {
    todo!();
}

fn main() {
    mut_ref() = 1;
    //~^ ERROR invalid left-hand side of assignment
    //~| HELP consider dereferencing here to assign a value to the left-hand side
}
