//@ run-flags:-Zrustdoc-scrape-examples
//@ compile-flags: --html-after-content extra.html
pub struct ObscurelyNamedType1;

impl ObscurelyNamedType1 {
    pub fn new() -> Self {
        ObscurelyNamedType1
    }
}
