fn main() {

    let y: int = 42;
    let mut x: int;
    loop {
        log(debug, y);
// tjc: not sure why it prints the same error twice
        while true { while true { while true { x = move y; copy x; } } }
        //~^ ERROR use of moved variable: `y`
        //~^^ NOTE move of variable occurred here
        //~^^^ ERROR use of moved variable: `y`
        //~^^^^ NOTE move of variable occurred here
    }
}
