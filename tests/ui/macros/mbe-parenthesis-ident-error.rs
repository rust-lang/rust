// Ensures MBEs with a invalid ident produce a readable error

macro_rules! (meepmeep) {
    //~^ ERROR: expected identifier, found `(`
    //~| HELP: try removing the parenthesis around the name for this `macro_rules!`
    () => {}
}
