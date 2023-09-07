fn main() {
    let mut test = Vec::new();
    let rofl: &Vec<Vec<i32>> = &mut test;
    //~^ HELP consider changing this binding's type
    rofl.push(Vec::new());
    //~^ ERROR cannot borrow `*rofl` as mutable, as it is behind an `&` reference
    //~| NOTE `rofl` is an `&` reference, so the data it refers to cannot be borrowed as mutable

    let mut mutvar = 42;
    let r = &mutvar;
    //~^ HELP consider changing this to be a mutable reference
    *r = 0;
    //~^ ERROR cannot assign to `*r`, which is behind an `&` reference
    //~| NOTE `r` is an `&` reference, so the data it refers to cannot be written

    #[rustfmt::skip]
    let x: &usize = &mut{0};
    //~^ HELP consider changing this binding's type
    *x = 1;
    //~^ ERROR cannot assign to `*x`, which is behind an `&` reference
    //~| NOTE `x` is an `&` reference, so the data it refers to cannot be written

    #[rustfmt::skip]
    let y: &usize = &mut(0);
    //~^ HELP consider changing this binding's type
    *y = 1;
    //~^ ERROR cannot assign to `*y`, which is behind an `&` reference
    //~| NOTE `y` is an `&` reference, so the data it refers to cannot be written
}
