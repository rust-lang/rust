// Note: it would be nice to give fewer warnings in these cases.

fn mutate_by_mut_ref(x: &mut uint) {
    *x = 0;
}

fn mutate_by_ref(&&x: uint) {
    //~^ WARNING unused variable: `x`
    x = 0; //~ ERROR assigning to argument
}

fn mutate_by_val(++x: uint) {
    //~^ WARNING unused variable: `x`
    x = 0; //~ ERROR assigning to argument
}

fn mutate_by_copy(+x: uint) {
    //~^ WARNING unused variable: `x`
    x = 0; //~ ERROR assigning to argument
    //~^ WARNING value assigned to `x` is never read
}

fn mutate_by_move(-x: uint) {
    //~^ WARNING unused variable: `x`
    x = 0; //~ ERROR assigning to argument
    //~^ WARNING value assigned to `x` is never read
}

fn main() {
}
