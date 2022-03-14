use std::io;

fn main(){
    let x: io::Result<()> = Ok(());
    match x {
        Err(ref e) if e.kind == io::EndOfFile {
            //~^ NOTE while parsing this struct
            //~| ERROR expected one of `!`, `.`, `::`, `=>`, `?`, or an operator, found `{`
            //~| NOTE expected one of `!`, `.`, `::`, `=>`, `?`, or an operator
            return
            //~^ ERROR expected identifier, found keyword `return`
            //~| NOTE expected identifier, found keyword
        }
        _ => {}
    }
}
