use std::io;

fn main(){
    let x: io::Result<()> = Ok(());
    match x {
        Err(ref e) if e.kind == io::EndOfFile {
        //~^ ERROR expected one of `!`, `.`, `::`, `=>`, `?`, or an operator, found `{`
            return
        }
        _ => {}
    }
}
