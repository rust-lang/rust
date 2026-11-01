// Verifies that the expected token errors for `extern crate` are raised.

extern "C" mod foo; //~ERROR expected `{`, found keyword `mod`
