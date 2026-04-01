A feature attribute named a feature that was disallowed in the compiler
command line flags.

Erroneous code example:

```ignore (can't specify compiler flags from doctests)
#![feature(never_type)] // error: the feature `never_type` is not in
                        // the list of allowed features
```

Delete the offending feature attribute, or add it to the list of allowed
features in the `-Z allow_features` flag.
