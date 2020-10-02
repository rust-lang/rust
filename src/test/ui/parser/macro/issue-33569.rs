macro_rules! foo {
    { $+ } => { //~ ERROR expected identifier, found `+`
                //~^ ERROR missing fragment specifier
        $(x)(y) //~ ERROR expected one of: `*`, `+`, or `?`
       //~^ ERROR attempted to repeat an expression containing no syntax variables
    }
}

foo!();

fn main() {}
