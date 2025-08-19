macro_rules! foo {
    { $+ } => { //~ ERROR expected identifier, found `+`
                //~^ ERROR missing fragment specifier
        $(x)(y) //~ ERROR expected one of: `*`, `+`, or `?`
    }
}

foo!(); //~ ERROR unexpected end

fn main() {}
