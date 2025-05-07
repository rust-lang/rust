//@ check-pass
//@ proc-macro: call-deprecated.rs

extern crate call_deprecated;

// These first two `#[allow(deprecated)]` attributes
// do nothing, since the AST nodes for `First` and `Second`
// haven't been assigned a `NodeId`.
// See #63221 for a discussion about how we should
// handle the interaction of 'inert' attributes and
// proc-macro attributes.

#[allow(deprecated)]
#[call_deprecated::attr] //~ WARN use of deprecated macro
struct First;

#[allow(deprecated)]
#[call_deprecated::attr_remove] //~ WARN use of deprecated macro
struct Second;

#[allow(deprecated)]
mod bar {
    #[allow(deprecated)]
    #[call_deprecated::attr]
    struct Third;

    #[allow(deprecated)]
    #[call_deprecated::attr_remove]
    struct Fourth;
}


fn main() {
}
