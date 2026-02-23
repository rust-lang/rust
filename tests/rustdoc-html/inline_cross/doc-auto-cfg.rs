// Test that `doc(auto_cfg)` works with inlined cross-crate re-exports.
//@ compile-flags: --cfg feature="extra" --cfg feature="addon"

#![feature(doc_cfg)]
#![crate_name = "it"]

//@ aux-build: doc-auto-cfg.rs
extern crate doc_auto_cfg;

// The cfg is on the reexported item.
// issue: <https://github.com/rust-lang/rust/issues/141301>
pub mod pre {
    //@ has 'it/pre/index.html' '//*[@class="stab portability"]' 'extension'
    //@ has 'it/pre/fn.compute.html' '//*[@class="stab portability"]' \
    //      'Available on extension only.'
    pub use doc_auto_cfg::*;

    // Indeed, this reexport doesn't have a cfg badge!
    // That's because this crate (`it`) wouldn't've compiled in the first place
    // if `--cfg extension` wasn't passed when compiling the auxiliary crate
    // contrary to the glob import above since `compute` wouldn't exist.
    //
    //@ !has 'it/pre/fn.calculate.html' '//*[@class="stab portability"]' \
    //      'Available on extension only.'
    pub use doc_auto_cfg::compute as calculate;

    // FIXME(HtmlDocCk): Ideally I would've used the following XPath here:
    // `*[@class="impl-items"][*[@id="method.transform"]]//*[@class="stab portability"]`
    //
    //@ has 'it/pre/struct.Kind.html' '//*[@id="method.transform"]' ''
    //@ has - '//*[@class="impl-items"]//*[@class="stab portability"]' \
    //        'Available on extension only.'
    pub use doc_auto_cfg::Type as Kind;
}

// The cfg is on the reexport.
pub mod post {
    // issue: <https://github.com/rust-lang/rust/issues/113982>
    //@ has 'it/post/index.html' '//*[@class="stab portability"]' 'extra'
    //@ has - '//*[@class="stab portability"]' 'extra and extension'
    //@ has 'it/post/struct.Type.html' '//*[@class="stab portability"]' \
    //      'Available on crate feature extra only.'
    //@ has 'it/post/fn.compute.html' '//*[@class="stab portability"]' \
    //      'Available on crate feature extra and extension only.'
    #[cfg(feature = "extra")]
    pub use doc_auto_cfg::*;

    //@ has 'it/post/index.html' '//*[@class="stab portability"]' 'addon'
    //@ has 'it/post/struct.Addon.html' '//*[@class="stab portability"]' \
    //      'Available on crate feature addon only.'
    #[cfg(feature = "addon")]
    pub use doc_auto_cfg::Type as Addon;
}
