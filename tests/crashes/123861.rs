//@ known-bug: #123861
//@ needs-rustc-debug-assertions

struct _;
fn mainIterator<_ = _> {}
