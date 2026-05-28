// Regression test for <https://github.com/rust-lang/rust/issues/100916>
//@ compile-flags: --cfg feature="routing"

#![crate_name = "foo"]
#![feature(doc_cfg)]

#[cfg(feature = "routing")]
pub mod routing {
    //@ has 'foo/macro.vpath.html' '//*[@class="stab portability"]' 'Available on crate feature routing only.'
    #[macro_export]
    macro_rules! vpath {
        () => {};
    }
}

#[doc(cfg(feature = "manual"))]
pub mod manual {
    //@ has 'foo/macro.manual_macro.html' '//*[@class="stab portability"]' 'Available on crate feature manual only.'
    #[macro_export]
    macro_rules! manual_macro {
        () => {};
    }
}

#[doc(cfg(feature = "outer"))]
pub mod outer {
    #[cfg(feature = "routing")]
    pub mod inner {
        //@ has 'foo/macro.nested_macro.html' '//*[@class="stab portability"]' 'Available on crate features outer and routing only.'
        #[macro_export]
        macro_rules! nested_macro {
            () => {};
        }
    }
}

#[cfg(feature = "routing")]
#[doc(auto_cfg = false)]
pub mod auto_cfg_disabled {
    //@ count 'foo/macro.no_auto_cfg_macro.html' '//*[@class="stab portability"]' 0
    #[macro_export]
    macro_rules! no_auto_cfg_macro {
        () => {};
    }
}

#[cfg(feature = "routing")]
#[doc(auto_cfg(hide(feature = "routing")))]
pub mod auto_cfg_hidden {
    //@ count 'foo/macro.hidden_cfg_macro.html' '//*[@class="stab portability"]' 0
    #[macro_export]
    macro_rules! hidden_cfg_macro {
        () => {};
    }
}

#[cfg(feature = "routing")]
#[doc(auto_cfg(hide(feature = "routing")))]
pub mod auto_cfg_shown {
    #[doc(auto_cfg(show(feature = "routing")))]
    pub mod inner {
        //@ has 'foo/macro.shown_cfg_macro.html' '//*[@class="stab portability"]' 'Available on crate feature routing only.'
        #[macro_export]
        macro_rules! shown_cfg_macro {
            () => {};
        }
    }
}
