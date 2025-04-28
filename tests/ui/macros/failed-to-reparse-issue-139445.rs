fn main() {
    assert_eq!(3, 'a,)
    //~^ ERROR expected `while`, `for`, `loop` or `{` after a label
    //~| ERROR expected `while`, `for`, `loop` or `{` after a label
    //~| ERROR expected expression, found ``
}
