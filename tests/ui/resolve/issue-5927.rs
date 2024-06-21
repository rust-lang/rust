fn main() {
    let z = match 3 {
        x(1) => x(1) //~ ERROR cannot find tuple struct or tuple variant `x`
        //~^ ERROR cannot find function `x`
    };
    assert!(z == 3);
}
