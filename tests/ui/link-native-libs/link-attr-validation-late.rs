#![feature(link_cfg)]

// Top-level ill-formed
#[link(name = "...", "literal")] //~ ERROR malformed `link` attribute input
#[link(name = "...", unknown)] //~ ERROR malformed `link` attribute input
extern "C" {}

// Duplicate arguments
#[link(name = "foo", name = "bar")] //~ ERROR malformed `link` attribute input
#[link(name = "...", kind = "dylib", kind = "bar")] //~ ERROR malformed `link` attribute input
#[link(name = "...", modifiers = "+verbatim", modifiers = "bar")] //~ ERROR malformed `link` attribute input
#[link(name = "...", cfg(false), cfg(false))] //~ ERROR malformed `link` attribute input
#[link(wasm_import_module = "foo", wasm_import_module = "bar")] //~ ERROR malformed `link` attribute input
extern "C" {}

// Ill-formed arguments
#[link(name)] //~ ERROR malformed `link` attribute input
#[link(name())] //~ ERROR malformed `link` attribute input
#[link(name = "...", kind)] //~ ERROR malformed `link` attribute input
#[link(name = "...", kind())] //~ ERROR malformed `link` attribute input
#[link(name = "...", modifiers)] //~ ERROR malformed `link` attribute input
#[link(name = "...", modifiers())] //~ ERROR malformed `link` attribute input
#[link(name = "...", cfg)] //~ ERROR malformed `link` attribute input
#[link(name = "...", cfg = "literal")] //~ ERROR malformed `link` attribute input
#[link(name = "...", cfg("literal"))] //~ ERROR `cfg` predicate key must be an identifier
#[link(name = "...", wasm_import_module)] //~ ERROR malformed `link` attribute input
#[link(name = "...", wasm_import_module())] //~ ERROR malformed `link` attribute input
extern "C" {}

// Basic modifier validation
#[link(name = "...", modifiers = "")] //~ ERROR invalid linking modifier syntax, expected '+' or '-' prefix
#[link(name = "...", modifiers = "no-plus-minus")] //~ ERROR invalid linking modifier syntax, expected '+' or '-' prefix
#[link(name = "...", modifiers = "+unknown")] //~ ERROR malformed `link` attribute input
#[link(name = "...", modifiers = "+verbatim,+verbatim")] //~ ERROR multiple `verbatim` modifiers
extern "C" {}

fn main() {}
