// rustfmt-stable: true

// cfg_select! doesn't get formatted, but indentation is updated.

#![crate_type = "lib"]
cfg_select! {
    _ => {
        fn foo(){} {}
    }
}

cfg_select! { _ => {{} {}}}
