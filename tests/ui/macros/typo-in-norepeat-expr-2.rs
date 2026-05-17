macro_rules! err {
    (begin $follow:ident end $arg:expr) => {
        [$arg]
    };
    (begin1 $arg1:ident end $agr2:expr) => {
        [$follow] //~ ERROR: cannot find macro parameter `$follow` in this scope
        //~^ NOTE: there is an macro metavariable with this name in another macro matcher
        //~| NOTE: expected expression
    };
}

macro_rules! err1 {
    (begin $follow:ident end $arg:expr) => {
        [$arg]
    };
    (begin1 $arg1:ident end) => {
        [$follo] //~ ERROR: cannot find macro parameter `$follo` in this scope
        //~| NOTE: expected expression
        //~| HELP: there is a macro metavariable with a similar name in another macro matcher
    };
}

macro_rules! err2 {
    (begin $follow:ident end $arg:expr) => {
        [$arg]
    };
    (begin1 $arg1:ident end) => {
        [$xyz] //~ ERROR: cannot find macro parameter `$xyz` in this scope
        //~^ NOTE: expected expression
        //~| NOTE available metavariable names are: $arg1
        //~| NOTE: macro metavariable not found
    };
}

fn main () {
    let _ = err![begin1 x  end ig]; //~ NOTE: in this expansion of err!
    let _ = err1![begin1 x  end]; //~ NOTE: in this expansion of err1!
            //~| NOTE: in this expansion of err1!

    let _ = err2![begin1 x  end]; //~ NOTE: in this expansion of err2!
            //~| NOTE in this expansion of err2!
}
