// ignore-tidy-cr
// check-pass

// This file checks the spans of intra-link warnings in a file with CRLF line endings. The
// .gitattributes file in this directory should enforce it.

/// [error]
pub struct A;
//~^^ WARNING `[error]` cannot be resolved

///
/// docs [error1]
//~^ WARNING `[error1]` cannot be resolved

/// docs [error2]
///
pub struct B;
//~^^^ WARNING `[error2]` cannot be resolved

/**
 * This is a multi-line comment.
 *
 * It also has an [error].
 */
pub struct C;
//~^^^ WARNING `[error]` cannot be resolved
