#![warn(clippy::doc_markdown)]

// Should only warn for the first line!
/// AviSynth documentation:
//~^ ERROR: item in documentation is missing backticks
///
/// > AvisynthPluginInit3 may be called more than once with different IScriptEnvironments.
///
/// <blockquote>bla AvisynthPluginInit3 bla</blockquote>
///
/// <q>bla AvisynthPluginInit3 bla</q>
pub struct Foo;
