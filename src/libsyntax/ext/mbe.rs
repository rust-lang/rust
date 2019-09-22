//! This module implements declarative macros: old `macro_rules` and the newer
//! `macro`. Declarative macros are also known as "macro by example", and that's
//! why we call this module `mbe`. For external documentation, prefer the
//! official terminology: "declarative macros".

crate mod transcribe;
crate mod macro_check;
crate mod macro_parser;
crate mod macro_rules;
crate mod quoted;
