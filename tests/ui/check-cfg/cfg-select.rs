//@ check-pass

#![feature(cfg_select)]
#![crate_type = "lib"]

cfg_select! {
    true => {}
    invalid_cfg1 => {}
    //~^ WARN unexpected `cfg` condition name
    _ => {}
}

cfg_select! {
    invalid_cfg2 => {}
    //~^ WARN unexpected `cfg` condition name
    true => {}
    _ => {}
}
