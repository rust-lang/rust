//@aux-build:proc_macro_attr.rs
//@compile-flags: --test --cfg dummy_cfg
#![feature(custom_inner_attributes)]
#![warn(clippy::mixed_attributes_style)]
#![allow(clippy::duplicated_attributes)]

#[macro_use]
extern crate proc_macro_attr;

#[allow(unused)] //~ ERROR: item has both inner and outer attributes
fn foo1() {
    #![allow(unused)]
}

#[allow(unused)]
#[allow(unused)]
fn foo2() {}

fn foo3() {
    #![allow(unused)]
    #![allow(unused)]
}

/// linux
//~^ ERROR: item has both inner and outer attributes
fn foo4() {
    //! windows
}

/// linux
/// windows
fn foo5() {}

fn foo6() {
    //! linux
    //! windows
}

#[allow(unused)] //~ ERROR: item has both inner and outer attributes
mod bar {
    #![allow(unused)]
}

fn main() {
    // test code goes here
}

// issue #12435
#[cfg(test)]
mod tests {
    //! Module doc, don't lint
}
#[allow(unused)]
mod baz {
    //! Module doc, don't lint
    const FOO: u8 = 0;
}
/// Module doc, don't lint
mod quz {
    #![allow(unused)]
}

mod issue_12530 {
    // don't lint different attributes entirely
    #[cfg(test)]
    mod tests {
        #![allow(clippy::unreadable_literal)]

        #[allow(dead_code)] //~ ERROR: item has both inner and outer attributes
        mod inner_mod {
            #![allow(dead_code)]
        }
    }
    #[cfg(dummy_cfg)]
    mod another_mod {
        #![allow(clippy::question_mark)]
    }
    /// Nested mod
    mod nested_mod {
        #[allow(dead_code)] //~ ERROR: item has both inner and outer attributes
        mod inner_mod {
            #![allow(dead_code)]
        }
    }
    /// Nested mod
    //~^ ERROR: item has both inner and outer attributes
    #[allow(unused)]
    mod nest_mod_2 {
        #![allow(unused)]

        #[allow(dead_code)] //~ ERROR: item has both inner and outer attributes
        mod inner_mod {
            #![allow(dead_code)]
        }
    }
    // Different path symbols - Known FN
    #[dummy]
    fn use_dummy() {
        #![proc_macro_attr::dummy]
    }
}
