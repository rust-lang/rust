% Pre-1.0 changes

### `std` facade

We should revisit some APIs in `libstd` now that the facade effort is complete.

For example, the treatment of environment variables in the new
`Command` API is waiting on access to hashtables before being
approved.

### Trait reform

Potential for standard conversion traits (`to`, `into`, `as`).

Probably many other opportunities here.

### Unboxed closures
