fn main() {

    let y: int = 42;
    let mut x: int;
    loop {
        log(debug, y);
        loop {
            loop {
                loop {
                    x <- y; //! ERROR use of moved variable
                    //!^ NOTE move of variable occurred here
                }
            }
        }
    }
}
