fn take(-_x: int) {}

fn main() {

    let x: int = 25;
    loop {
        take(move x); //~ ERROR use of moved variable: `x`
        //~^ NOTE move of variable occurred here
    }
}
