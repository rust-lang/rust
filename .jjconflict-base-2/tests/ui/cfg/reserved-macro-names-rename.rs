//@edition:2018

#![crate_type = "lib"]


mod a {
    use ignore as cfg;
    //~^ERROR name `cfg` is reserved in attribute namespace
}

mod b {
    use cfg_attr as cfg;
    //~^ERROR name `cfg` is reserved in attribute namespace
}

mod c {
    use cfg as cfg;
    //~^ERROR `cfg` is ambiguous
}

mod d {
    use inline as cfg_attr;
    //~^ERROR name `cfg_attr` is reserved in attribute namespace
}

mod e {
    use not_found as cfg; // trigger "unresolved import", not "cfg reserved".
    //~^ ERROR unresolved import `not_found`
}
