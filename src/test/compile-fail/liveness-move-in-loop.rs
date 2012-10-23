fn main() {

    let y: int = 42;
    let mut x: int;
    loop {
        log(debug, y);
        loop {
            loop {
                loop {
// tjc: Not sure why it prints the same error twice
                    x = move y; //~ ERROR use of moved variable
                    //~^ NOTE move of variable occurred here
                    //~^^ ERROR use of moved variable
                    //~^^^ NOTE move of variable occurred here

                    copy x;
                }
            }
        }
    }
}
