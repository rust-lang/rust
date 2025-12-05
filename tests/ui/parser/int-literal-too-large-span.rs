// issue #17123

fn main() {
    9999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999
    //~^ ERROR integer literal is too large
        ; // the span shouldn't point to this.
}
