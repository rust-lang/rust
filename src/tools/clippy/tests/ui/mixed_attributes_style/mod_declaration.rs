//@error-in-other-file: item has both inner and outer attributes
//@no-rustfix
#[path = "auxiliary/submodule.rs"] // don't lint.
/// This doc comment should not lint, it could be used to add context to the original module doc
mod submodule;
