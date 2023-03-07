fn main() {
    match () {
        [()] => { }
        //~^ ERROR expected an array or slice, found `()`
    }
}
