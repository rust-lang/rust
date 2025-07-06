#![feature(cfg_select)]
#![crate_type = "lib"]

fn print() {
    println!(cfg_select! {
        unix => { "unix" }
        _ => { "not unix" }
    });
}

fn arm_rhs_must_be_in_braces() -> i32 {
    cfg_select! {
        true => 1
        //~^ ERROR: expected `{`, found `1`
    }
}

cfg_select! {
    _ => {}
    true => {}
    //~^ WARN unreachable rule
}

cfg_select! {
    //~^ ERROR none of the rules in this `cfg_select` evaluated to true
    false => {}
}
