fn main() {
    let a = &mut 0;
    let b = 1;
    let _ = a < b;
    //~^ ERROR mismatched types
    //~| HELP consider dereferencing here
    //~| SUGGESTION *
}
