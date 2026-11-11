fn main() {
    let x = |x|
        let y = x; //~ ERROR expected expression, found `let` statement
        let _ = () + ();
        y
}
