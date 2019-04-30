// exec-env:RUSTC_LOG=std::ptr

// In issue #9487, it was realized that std::ptr was invoking the logging
// infrastructure, and when std::ptr was used during runtime initialization,
// this caused some serious problems. The problems have since been fixed, but
// this test will trigger "output during runtime initialization" to make sure
// that the bug isn't re-introduced.

// pretty-expanded FIXME #23616

pub fn main() {}
