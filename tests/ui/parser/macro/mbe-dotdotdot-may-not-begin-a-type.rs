// A bare `...` represents `CVarArgs` (`VaListImpl<'_>`) in function argument type
// position without being a proper type syntactically.
// This test ensures that we do not regress certain MBE calls would we ever promote
// `...` to a proper type syntactically.

//@ check-pass

macro_rules! ck { ($ty:ty) => { compile_error!(""); }; (...) => {}; }
ck!(...);

fn main() {}
