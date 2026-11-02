//@ rustfmt-unstable: true
//@ rustfmt-show_parse_errors: false
// There are parse errors that we don't wont to show in test output

cfg_select! {
    unix   =>   {   fn   bar() {} }
}

cfg_select! { unix => { fn foo {} } }
