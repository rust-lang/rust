#![feature(doc_cfg)]

//@ is "$.index[?(@.name=='f')].attrs" '[{"other": "#[doc(auto_cfg(hide(bar, values(none())), hide(blob, values(\"a\", \"14\"))))]"}]'
#[doc(auto_cfg(hide(bar, values(none())), hide(blob, values("a", "14")),))]
pub fn f() {}
