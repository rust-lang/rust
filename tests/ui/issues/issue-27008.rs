struct S;

fn main() {
    let b = [0; S];
    //~^ ERROR the constant `S` is not of type `usize`
}
