#![warn(clippy::multi_assignments)]
fn main() {
    let (mut a, mut b, mut c, mut d) = ((), (), (), ());
    a = b = c;
    //~^ multi_assignments

    a = b = c = d;
    //~^ multi_assignments
    //~| multi_assignments

    a = b = { c };
    //~^ multi_assignments

    a = { b = c };
    //~^ multi_assignments

    a = (b = c);
    //~^ multi_assignments
}
