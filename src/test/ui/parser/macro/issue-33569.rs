macro_rules! foo {
    { $+ } => { //~ ERROR expected identifier, found `+`
                //~^ ERROR missing fragment specifier
        $(x)(y) //~ ERROR expected `*` or `+`
    }
}

foo!();
