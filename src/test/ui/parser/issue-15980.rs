use std::io;

fn main(){
    let x: io::Result<()> = Ok(());
    match x {
        Err(ref e) if e.kind == io::EndOfFile {
            //~^ NOTE while parsing this struct
            return
            //~^ ERROR expected identifier, found keyword `return`
            //~| NOTE expected identifier, found keyword
        }
        //~^ NOTE expected one of `.`, `=>`, `?`, or an operator
        _ => {}
        //~^ ERROR expected one of `.`, `=>`, `?`, or an operator, found reserved identifier `_`
        //~| NOTE unexpected token
    }
}
