//@ compile-flags: --document-private-items

#![feature(decl_macro)]

//@ has decl_macro/macro.my_macro.html //pre 'pub macro my_macro() {'
//@ has - //pre '...'
//@ has - //pre '}'
pub macro my_macro() {

}

//@ has decl_macro/macro.my_macro_2.html //pre 'pub macro my_macro_2($($tok:tt)*) {'
//@ has - //pre '...'
//@ has - //pre '}'
pub macro my_macro_2($($tok:tt)*) {

}

//@ has decl_macro/macro.my_macro_multi.html //pre 'pub macro my_macro_multi {'
//@ has - //pre '(_) => { ... },'
//@ has - //pre '($foo:ident . $bar:expr) => { ... },'
//@ has - //pre '($($foo:literal),+) => { ... },'
//@ has - //pre '}'
pub macro my_macro_multi {
    (_) => {

    },
    ($foo:ident . $bar:expr) => {

    },
    ($($foo:literal),+) => {

    }
}

//@ has decl_macro/macro.by_example_single.html //pre 'pub macro by_example_single($foo:expr) {'
//@ has - //pre '...'
//@ has - //pre '}'
pub macro by_example_single {
    ($foo:expr) => {}
}

mod a {
    mod b {
        //@ has decl_macro/a/b/macro.by_example_vis.html //pre 'pub(super) macro by_example_vis($foo:expr) {'
        pub(in super) macro by_example_vis {
            ($foo:expr) => {}
        }
        mod c {
            //@ has decl_macro/a/b/c/macro.by_example_vis_named.html //pre 'pub(in a) macro by_example_vis_named($foo:expr) {'
            // Regression test for <https://github.com/rust-lang/rust/issues/83000>:
            //@ has - '//pre[@class="rust item-decl"]//a[@class="mod"]/@href' '../../index.html'
            pub(in a) macro by_example_vis_named {
                ($foo:expr) => {}
            }
        }
    }
}
