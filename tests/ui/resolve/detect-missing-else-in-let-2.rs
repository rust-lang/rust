// the compiler currently will suggest the user to add `else` in this case
// it shouldn't because `bar` is an irrefutable pattern, the else block would never be valid/useful
// it is currently unmitigated
fn main(){
    let foo = 12;
    let bar = foo{return;};//~ ERROR E0574
    //~| HELP you might have meant to write a diverging block
    //~| HELP escape `return` to use it as an identifier
    //~| ERROR expected identifier, found keyword `return`
}
