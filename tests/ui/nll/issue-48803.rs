fn flatten<'a, 'b, T>(x: &'a &'b T) -> &'a T {
    x
}

fn main() {
    let mut x = "original";
    let y = &x;
    let z = &y;
    let w = flatten(z);
    x = "modified";
    //~^ ERROR cannot assign to `x` because it is borrowed [E0506]
    println!("{}", w); // prints "modified"
}
