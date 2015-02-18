% Organization [FIXME: needs RFC]

> **[FIXME]** What else?

### Reexport the most important types at the crate level.

Crates `pub use` the most common types for convenience, so that clients do not
have to remember or write the crate's module hierarchy to use these types.

### Define types and operations together.

Type definitions and the functions/methods that operate on them should be
defined together in a single module, with the type appearing above the
functions/methods.
