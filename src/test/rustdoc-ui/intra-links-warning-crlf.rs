// ignore-tidy-cr

// build-pass (FIXME(62277): could be check-pass?)

// This file checks the spans of intra-link warnings in a file with CRLF line endings. The
// .gitattributes file in this directory should enforce it.

/// [error]
pub struct A;

///
/// docs [error1]

/// docs [error2]
///
pub struct B;

/**
 * This is a multi-line comment.
 *
 * It also has an [error].
 */
pub struct C;
