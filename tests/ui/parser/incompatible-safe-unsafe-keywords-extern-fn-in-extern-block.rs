//! Check that we emit a targeted suggestion for an extern fn that uses incompatible `safe` and
//! `unsafe` keywords.
#![crate_type = "lib"]

unsafe extern {
//~^ NOTE while parsing this item list starting here
    pub safe unsafe extern fn foo() {}
    //~^ ERROR expected one of `extern` or `fn`, found keyword `unsafe`
    //~| NOTE expected one of `extern` or `fn`
    //~| `safe` and `unsafe` are incompatible, use only one of the keywords
} //~ NOTE the item list ends here
