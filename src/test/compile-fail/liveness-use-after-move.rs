fn main() {
    let x = @5;
    let y <- x; //~ NOTE move of variable occurred here
    log(debug, *x); //~ ERROR use of moved variable: `x`
    copy y;
}
