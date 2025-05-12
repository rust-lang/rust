// Checks that undocumented private macros will not generate `missing_docs`
// lints, but public ones will.
//
// This is a regression test for issue #57569
#![deny(missing_docs)]
#![feature(decl_macro)]
//! Empty documentation.

macro new_style_private_macro {
    () => ()
}

pub(crate) macro new_style_crate_macro {
    () => ()
}

macro_rules! old_style_private_macro {
    () => ()
}

mod submodule {
    pub macro new_style_macro_in_private_module {
        () => ()
    }

    macro_rules! old_style_mod_private_macro {
        () => ()
    }

    #[macro_export]
    macro_rules! exported_to_top_level {
        //~^ ERROR missing documentation for a macro
        () => ()
    }
}

pub macro top_level_pub_macro {
    //~^ ERROR missing documentation for a macro
    () => ()
}

/// Empty documentation.
pub fn main() {}
