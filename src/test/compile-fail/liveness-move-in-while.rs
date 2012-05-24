fn main() {

    let y: int = 42;
    let mut x: int;
    loop {
        log(debug, y);
        while true { while true { while true { x <- y; copy x; } } }
        //!^ ERROR use of moved variable: `y`
        //!^^ NOTE move of variable occurred here
    }
}
