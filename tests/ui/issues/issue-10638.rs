// run-pass
// pretty-expanded FIXME #23616

pub fn main() {
    //// I am not a doc comment!
    ////////////////// still not a doc comment
    /////**** nope, me neither */
    /*** And neither am I! */
    5;
    /*****! certainly not I */
}
