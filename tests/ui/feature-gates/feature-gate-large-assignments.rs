// check that `move_size_limit` is feature-gated

#![move_size_limit = "42"] //~ ERROR the `#[move_size_limit]` attribute is an experimental feature

fn main() {}
