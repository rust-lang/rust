#![warn(clippy::nonminimal_bool)]
//@no-rustfix

fn issue_13436() {
    let opt_opt = Some(Some(500));
    _ = !opt_opt.is_some_and(|x| !x.is_some_and(|y| y != 1000)); //~ ERROR: this boolean expression can be simplified
}

fn main() {}
