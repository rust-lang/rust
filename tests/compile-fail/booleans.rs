#![feature(plugin)]
#![plugin(clippy)]
#![deny(nonminimal_bool)]

#[allow(unused)]
fn main() {
    let a: bool = unimplemented!();
    let b: bool = unimplemented!();
    let _ = a && b || a; //~ ERROR this boolean expression can be simplified
    //|~ HELP for further information visit
    //|~ SUGGESTION let _ = a;
    let _ = !(a && b); //~ ERROR this boolean expression can be simplified
    //|~ HELP for further information visit
    //|~ SUGGESTION let _ = !b || !a;
    let _ = !true; //~ ERROR this boolean expression can be simplified
    //|~ HELP for further information visit
    //|~ SUGGESTION let _ = false;
    let _ = !false; //~ ERROR this boolean expression can be simplified
    //|~ HELP for further information visit
    //|~ SUGGESTION let _ = true;
    let _ = !!a; //~ ERROR this boolean expression can be simplified
    //|~ HELP for further information visit
    //|~ SUGGESTION let _ = a;
}
