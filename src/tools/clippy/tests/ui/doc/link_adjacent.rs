#![warn(clippy::doc_link_code)]

//! Test case for code links that are adjacent to code text.
//!
//! This is not an example: `first``second`
//!
//! Neither is this: [`first`](x)
//!
//! Neither is this: [`first`](x) `second`
//!
//! Neither is this: [first](x)`second`
//!
//! This is: [`first`](x)`second`
//~^ ERROR: adjacent
//!
//! So is this `first`[`second`](x)
//~^ ERROR: adjacent
//!
//! So is this [`first`](x)[`second`](x)
//~^ ERROR: adjacent
//!
//! So is this [`first`](x)[`second`](x)[`third`](x)
//~^ ERROR: adjacent
//!
//! So is this [`first`](x)`second`[`third`](x)
//~^ ERROR: adjacent

/// Test case for code links that are adjacent to code text.
///
/// This is not an example: `first``second` arst
///
/// Neither is this: [`first`](x) arst
///
/// Neither is this: [`first`](x) `second` arst
///
/// Neither is this: [first](x)`second` arst
///
/// This is: [`first`](x)`second` arst
//~^ ERROR: adjacent
///
/// So is this `first`[`second`](x) arst
//~^ ERROR: adjacent
///
/// So is this [`first`](x)[`second`](x) arst
//~^ ERROR: adjacent
///
/// So is this [`first`](x)[`second`](x)[`third`](x) arst
//~^ ERROR: adjacent
///
/// So is this [`first`](x)`second`[`third`](x) arst
//~^ ERROR: adjacent
pub struct WithTrailing;
