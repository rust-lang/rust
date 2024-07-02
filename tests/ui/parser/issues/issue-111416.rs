fn main() {
    let my = monad_bind(mx, T: Try);
    //~^ ERROR expected identifier, found `:`
    //~| ERROR cannot find value `mx`
    //~| ERROR cannot find value `Try`
    //~| ERROR cannot find function `monad_bind`
}
