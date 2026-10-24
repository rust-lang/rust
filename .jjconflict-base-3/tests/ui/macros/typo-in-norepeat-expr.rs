//@ run-rustfix
macro_rules! m {
    (begin $ard:ident end) => {
        [$arg] //~ ERROR: cannot find macro parameter `$arg` in this scope
        //~^ NOTE: not found in this scope
        //~| HELP: there is a macro metavariable with a similar name
    };
}

fn main() {
    let x = 1;
    let _ = m![begin x end]; //~ NOTE: in this expansion of m!
    //~| NOTE: in this expansion of m!
}
