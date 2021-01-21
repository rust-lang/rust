#![warn(unused_labels)]

fn main() {
    'a: for _ in 0..1 {
        break 'a;
    }
    'b: for _ in 0..1 {
        //~^ WARN unused label
        break b; //~ ERROR cannot find value `b` in this scope
    }
    c: for _ in 0..1 { //~ ERROR expected identifier, found keyword `for`
        //~^ ERROR expected `<`, found reserved identifier `_`
        break 'c;
    }
    d: for _ in 0..1 {
        break ;
    }
}
