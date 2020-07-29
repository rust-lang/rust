// Rustdoc would previously report resolution failures on items that weren't in the public docs.
// These failures were legitimate, but not truly relevant - the docs in question couldn't be
// checked for accuracy anyway.

#![deny(intra_doc_resolution_failures)]

/// ooh, i'm a [rebel] just for kicks
struct SomeStruct;
