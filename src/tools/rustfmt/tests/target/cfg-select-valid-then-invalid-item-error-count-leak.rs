//@ rustfmt-unstable: true
cfg_select! {
    unix => {
        fn bar() {}
    }
}

cfg_select! { unix => { fn foo {} } }
