fn main() {
    let y = 1;
    match y {
       a | b => {} //~ ERROR variable `a` from pattern #1 is not bound in pattern #2
       //~^ ERROR variable `b` from pattern #2 is not bound in pattern #1
    }
}
