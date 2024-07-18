//@ revisions: gated ungated

// Very pared down version of the same named test on nightly, since we only want
// to validate that `unsafe static` is not being accidentally accepted by the parser.

unsafe static LOL: u8 = 0;
//~^ ERROR: static items cannot be declared with `unsafe` safety qualifier outside of `extern` block

fn main() {}
