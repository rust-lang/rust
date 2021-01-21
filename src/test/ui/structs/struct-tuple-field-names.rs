struct S(i32, f32);
enum E {
    S(i32, f32),
}
fn main() {
    let x = E::S(1, 2.2);
    match x {
        E::S { 0, 1 } => {}
        //~^ ERROR tuple variant `E::S` uses a bare index in a struct pattern [E0769]
    }
    let y = S(1, 2.2);
    match y {
        S { } => {} //~ ERROR: tuple variant `S` written as struct variant [E0769]
    }
}
