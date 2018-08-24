fn main() {
    let y = 1;
    match y {
       a | b => {} //~  ERROR variable `a` is not bound in all patterns
                   //~^ ERROR variable `b` is not bound in all patterns
    }
}
