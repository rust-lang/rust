// https://github.com/rust-lang/rust/issues/72340

#![crate_name = "foo"]

pub struct Body;

impl Body {
    pub fn empty() -> Self {
        Body
    }

}

impl Default for Body {
    //@ has foo/struct.Body.html '//a/@href' 'struct.Body.html#method.empty'

    /// Returns [`Body::empty()`](Body::empty).
    fn default() -> Body {
        Body::empty()
    }
}
