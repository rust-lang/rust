fn take(-x: int) {}

fn main() {

    let x: int = 25;
    loop {
        take(x); //! ERROR use of moved variable: `x`
        //!^ NOTE move of variable occurred here
    }
}
