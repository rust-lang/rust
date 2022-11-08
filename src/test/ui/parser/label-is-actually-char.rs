fn main() {
    let c = 'a;
    //~^ ERROR expected `while`, `for`, `loop` or `{` after a label
    //~| HELP add `'` to close the char literal
    match c {
        'a'..='b => {}
        //~^ ERROR unexpected token: `'b`
        //~| HELP add `'` to close the char literal
        _ => {}
    }
    let x = ['a, 'b];
    //~^ ERROR expected `while`, `for`, `loop` or `{` after a label
    //~| ERROR expected `while`, `for`, `loop` or `{` after a label
    //~| HELP add `'` to close the char literal
    //~| HELP add `'` to close the char literal
}
