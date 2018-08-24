fn func((1, (Some(1), 2..=3)): (isize, (Option<isize>, isize))) { }
//~^ ERROR refutable pattern in function argument: `(_, _)` not covered

fn main() {
    let (1, (Some(1), 2..=3)) = (1, (None, 2));
    //~^ ERROR refutable pattern in local binding: `(_, _)` not covered
}
