// ignore-tidy-cr
//@ check-pass

// This file checks the spans of intra-link warnings in a file with CRLF line endings. The
// .gitattributes file in this directory should enforce it.

/// [error]
pub struct A;
//~^^ WARNING `error`

///
/// docs [error1]
//~^ WARNING `error1`

/// docs [error2]
///
pub struct B;
//~^^^ WARNING `error2`

/**
 * This is a multi-line comment.
 *
 * It also has an [error].
 */
pub struct C;
//~^^^ WARNING `error`
