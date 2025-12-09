#![warn(unreachable_cfg_select_predicates)]
//~^ WARN unknown lint: `unreachable_cfg_select_predicates`

cfg_select! {
    //~^ ERROR use of unstable library feature `cfg_select`
    _ => {}
    // With the feature enabled, this branch would trip the unreachable_cfg_select_predicate lint.
    true => {}
}

fn main() {}
