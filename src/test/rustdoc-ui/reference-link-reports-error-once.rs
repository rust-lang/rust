#![deny(broken_intra_doc_links)]

/// Links to [a] [link][a]
/// And also a [third link][a]
/// And also a [reference link][b]
///
/// Other links to the same target should still emit error: [ref] //~ERROR unresolved link to `ref`
/// Duplicate [ref] //~ERROR unresolved link to `ref`
///
/// Other links to other targets should still emit error: [ref2] //~ERROR unresolved link to `ref2`
/// Duplicate [ref2] //~ERROR unresolved link to `ref2`
///
/// [a]: ref
//~^ ERROR unresolved link to `ref`
/// [b]: ref2
//~^ ERROR unresolved link to

/// [ref][]
//~^ ERROR unresolved link
pub fn f() {}
