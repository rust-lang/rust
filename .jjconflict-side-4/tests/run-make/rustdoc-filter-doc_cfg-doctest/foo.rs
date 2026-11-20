#![feature(doc_cfg)]

/// ```
/// assert!(true);
/// ```
#[doc(cfg(spec))]
fn f() {}

#[doc(cfg(false))]
mod dummy {
    /// ```
    /// assert!(true);
    /// ```
    fn f2() {}
}
