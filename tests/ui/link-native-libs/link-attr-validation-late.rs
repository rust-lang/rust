#![feature(link_cfg)]

// Top-level ill-formed
#[link(name = "...", "literal")] //~ ERROR unexpected `#[link]` argument
#[link(name = "...", unknown)] //~ ERROR unexpected `#[link]` argument
extern "C" {}

// Duplicate arguments
#[link(name = "foo", name = "bar")] //~ ERROR multiple `name` arguments
#[link(name = "...", kind = "dylib", kind = "bar")] //~ ERROR multiple `kind` arguments
#[link(name = "...", modifiers = "+verbatim", modifiers = "bar")] //~ ERROR multiple `modifiers` arguments
#[link(name = "...", cfg(false), cfg(false))] //~ ERROR multiple `cfg` arguments
#[link(wasm_import_module = "foo", wasm_import_module = "bar")] //~ ERROR multiple `wasm_import_module` arguments
extern "C" {}

// Ill-formed arguments
#[link(name)] //~ ERROR link name must be of the form `name = "string"`
              //~| ERROR `#[link]` attribute requires a `name = "string"` argument
#[link(name())] //~ ERROR link name must be of the form `name = "string"`
              //~| ERROR `#[link]` attribute requires a `name = "string"` argument
#[link(name = "...", kind)] //~ ERROR link kind must be of the form `kind = "string"`
#[link(name = "...", kind())] //~ ERROR link kind must be of the form `kind = "string"`
#[link(name = "...", modifiers)] //~ ERROR link modifiers must be of the form `modifiers = "string"`
#[link(name = "...", modifiers())] //~ ERROR link modifiers must be of the form `modifiers = "string"`
#[link(name = "...", cfg)] //~ ERROR link cfg must be of the form `cfg(/* predicate */)`
#[link(name = "...", cfg = "literal")] //~ ERROR link cfg must be of the form `cfg(/* predicate */)`
#[link(name = "...", cfg("literal"))] //~ ERROR link cfg must have a single predicate argument
#[link(name = "...", wasm_import_module)] //~ ERROR wasm import module must be of the form `wasm_import_module = "string"`
#[link(name = "...", wasm_import_module())] //~ ERROR wasm import module must be of the form `wasm_import_module = "string"`
extern "C" {}

// Basic modifier validation
#[link(name = "...", modifiers = "")] //~ ERROR invalid linking modifier syntax, expected '+' or '-' prefix
#[link(name = "...", modifiers = "no-plus-minus")] //~ ERROR invalid linking modifier syntax, expected '+' or '-' prefix
#[link(name = "...", modifiers = "+unknown")] //~ ERROR unknown linking modifier `unknown`
#[link(name = "...", modifiers = "+verbatim,+verbatim")] //~ ERROR multiple `verbatim` modifiers
extern "C" {}

fn main() {}
