//@ check-pass

#![crate_type = "lib"]

cfg_select! {
    false => {}
    invalid_cfg1 => {}
    //~^ WARN unexpected `cfg` condition name
    _ => {}
}

cfg_select! {
    invalid_cfg2 => {}
    //~^ WARN unexpected `cfg` condition name
    false => {}
    _ => {}
}
