fn foo(+x: ~int) -> int {
    let y = &*x; //~ NOTE loan of argument granted here
    free(move x); //~ ERROR moving out of argument prohibited due to outstanding loan
    *y
}

fn free(+_x: ~int) {
}

fn main() {
}