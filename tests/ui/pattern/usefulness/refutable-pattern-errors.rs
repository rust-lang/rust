//@ dont-require-annotations: NOTE

fn func((1, (Some(1), 2..=3)): (isize, (Option<isize>, isize))) {}
//~^ ERROR refutable pattern in function argument
//~| NOTE `(..=0_isize, _)` and `(2_isize.., _)` not covered

fn main() {
    let (1, (Some(1), 2..=3)) = (1, (None, 2));
    //~^ ERROR refutable pattern in local binding
    //~| NOTE `(i32::MIN..=0_i32, _)` and `(2_i32..=i32::MAX, _)` not covered
}
