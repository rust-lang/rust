// Test for issue #14581.

fn f((a, a): (isize, isize)) {} //~ ERROR identifier `a` is bound more than once

fn main() {
    let (a, a) = (1, 1);    //~ ERROR identifier `a` is bound more than once
}
