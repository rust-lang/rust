#![deny(broken_intra_doc_links)]
#![feature(intra_doc_pointers)]
// These are links that could reasonably expected to work, but don't.

// `[]` isn't supported because it had too many false positives.
//! [X]([T]::not_here)
//! [Y](&[]::not_here)
//! [X]([]::not_here)
//! [Y]([T;N]::not_here)

// These don't work because markdown syntax doesn't allow it.
//! [[T]::rotate_left] //~ ERROR unresolved link to `T`
//! [&[]::not_here]
//![Z]([T; N]::map) //~ ERROR unresolved link to `Z`
//! [`[T; N]::map`]
//! [[]::map]
//! [Z][] //~ ERROR unresolved link to `Z`
//!
//! [Z]: [T; N]::map //~ ERROR unresolved link to `Z`

// `()` isn't supported because it had too many false positives.
//! [()::not_here]
//! [X]((,)::not_here)
//! [(,)::not_here]

// FIXME: Associated items on some primitives aren't working, because the impls
// are part of the compiler instead of being part of the source code.
//! [unit::eq] //~ ERROR unresolved
//! [tuple::eq] //~ ERROR unresolved
//! [fn::eq] //~ ERROR unresolved
//! [never::eq] //~ ERROR unresolved

// FIXME(#78800): This breaks because it's a blanket impl
// (I think? Might break for other reasons too.)
//! [reference::deref] //~ ERROR unresolved
