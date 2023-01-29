fn main() {
    let x = 42;
    match x {
        0..=73 => {},
        74..=> {},   //~ ERROR unexpected `=>` after open range
                     //~^ ERROR expected one of `=>`, `if`, or `|`, found `>`
    }
}
