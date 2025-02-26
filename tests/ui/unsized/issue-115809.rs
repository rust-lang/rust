//@ compile-flags: --emit=link -Zmir-opt-level=2 -Zvalidate-mir

fn foo<T>() {
    let a: [i32; 0] = [];
    match [a[..]] {
        //~^ ERROR cannot move a value of type `[i32]
        //~| ERROR cannot move out of type `[i32]`, a non-copy slice
        [[x]] => {}
        _ => (),
    }
}

fn main() {}
