# `min-recursion-limit`

This flag sets a minimum recursion limit for the compiler. The final recursion limit is calculated as `max(min_recursion_limit, recursion_limit_from_crate)`. This cannot ever lower the recursion limit. Unless the current crate has an explicitly low `recursion_limit` attribute, any value less than the current default does not have an effect.

The recursion limit affects (among other things):

- macro expansion
- the trait solver
- const evaluation
- query depth

This flag is particularly useful when using the next trait solver (`-Z next-solver`), which may require a higher recursion limit for crates that were compiled successfully with the old solver.
