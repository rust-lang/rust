// detect missing else in let statement. (issue #135857)
fn main() {
    let foo = Some(1);
    let Some(a) = foo{return;}; //~ ERROR E0574
    //~| HELP try adding `else` here:
    //~| HELP escape `return` to use it as an identifier
    //~| ERROR expected identifier, found keyword `return`
    let Some(b) = foo{todo!()}; //~ ERROR E0574
    //~| HELP try adding `else` here:
    //~| ERROR expected one of `,`, `:`, or `}`, found `!`
}
