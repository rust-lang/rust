macro_rules! foo {
    { $+ } => { //~ ERROR expected identifier, found `+`
                //~^ ERROR missing fragment specifier
                //~| ERROR missing fragment specifier
                //~| WARN this was previously accepted
        $(x)(y) //~ ERROR expected one of: `*`, `+`, or `?`
    }
}

foo!();

fn main() {}
