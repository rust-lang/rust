
#[legacy_exports]
mod lexer;
#[legacy_exports]
mod parser;
#[legacy_exports]
mod token;
#[legacy_exports]
mod comments;
#[legacy_exports]
mod attr;
#[legacy_exports]

/// Common routines shared by parser mods
#[legacy_exports]
mod common;

/// Functions dealing with operator precedence
#[legacy_exports]
mod prec;

/// Routines the parser uses to classify AST nodes
#[legacy_exports]
mod classify;

/// Reporting obsolete syntax
#[legacy_exports]
mod obsolete;
