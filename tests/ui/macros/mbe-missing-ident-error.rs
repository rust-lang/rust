// Ensures MBEs with a missing ident produce a readable error

macro_rules! {
    //~^ ERROR: expected identifier, found `{`
    //~| HELP: maybe you have forgotten to define a name for this `macro_rules!`
    () => {}
}
