#![feature(coverage_attribute)]
//@ edition: 2024
//@ revisions: none one two
//@[one] ignore-coverage-map
//@[two] ignore-coverage-map

// Basic test for if-let chains.

fn if_let_chain(opt_opt_msg: Option<Option<&str>>) {
    if let Some(opt_msg) = opt_opt_msg
        && let Some(msg) = opt_msg
    {
        say(msg);
    }

    if let Some(opt_msg) = opt_opt_msg
        && let Some(msg) = opt_msg
    {
        say(msg)
    }

    say("goodbye");
}

#[coverage(off)]
fn main() {
    let opt_opt_msg = cfg_select!(
        none => None,
        one => Some(None),
        two => Some(Some("hello")),
    );
    if_let_chain(opt_opt_msg);
}

#[coverage(off)]
fn say(msg: &str) {
    println!("{msg}");
}
