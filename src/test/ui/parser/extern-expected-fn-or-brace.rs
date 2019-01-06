// Verifies that the expected token errors for `extern crate` are
// raised

extern "C" mod foo; //~ERROR expected one of `fn` or `{`, found `mod`
