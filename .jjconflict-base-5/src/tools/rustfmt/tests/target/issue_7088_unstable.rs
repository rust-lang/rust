// rustfmt-unstable: true

#![crate_type = "lib"]
cfg_select! {
    _ => {
        fn foo() {}
        {}
    }
}

cfg_select! {
    _ => {
        {}
        {}
    }
}
