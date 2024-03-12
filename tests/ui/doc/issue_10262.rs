#![warn(clippy::doc_markdown)]

// Should only warn for the first line!
/// AviSynth documentation:
//~^ ERROR: item in documentation is missing backticks
///
/// > AvisynthPluginInit3 may be called more than once with different IScriptEnvironments.
pub struct Foo;
