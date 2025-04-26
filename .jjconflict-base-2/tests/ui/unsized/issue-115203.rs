//@ compile-flags: --emit link

fn main() {
    let a: [i32; 0] = [];
    match [a[..]] {
        //~^ ERROR cannot move a value of type `[i32]
        //~| ERROR cannot move out of type `[i32]`, a non-copy slice
        [[]] => (),
        _ => (),
    }
}
