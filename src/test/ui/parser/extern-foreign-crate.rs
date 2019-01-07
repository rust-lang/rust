// Verifies that the expected token errors for `extern crate` are
// raised

extern crate foo {} //~ERROR expected one of `;` or `as`, found `{`
