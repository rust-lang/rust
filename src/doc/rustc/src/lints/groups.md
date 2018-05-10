# Lint Groups

`rustc` has the concept of a "lint group", where you can toggle several warnings
through one name.

For example, the `nonstandard-style` lint sets `non-camel-case-types`,
`non-snake-case`, and `non-upper-case-globals` all at once. So these are
equivalent:

```bash
$ rustc -D nonstandard-style
$ rustc -D non-camel-case-types -D non-snake-case -D non-upper-case-globals
```

Here's a list of each lint group, and the lints that they are made up of:

| group               | description                                                   | lints                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
|---------------------|---------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| nonstandard-style   | Violation of standard naming conventions                      | non-camel-case-types, non-snake-case, non-upper-case-globals                                                                                                                                                                                                                                                                                                                                                                                                                           |
| warnings            | all lints that would be issuing warnings                      | all lints that would be issuing warnings                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| edition-2018        | Lints that will be turned into errors in Rust 2018            | tyvar-behind-raw-pointer                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| rust-2018-idioms    | Lints to nudge you toward idiomatic features of Rust 2018     | bare-trait-object, unreachable-pub                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| unused              | These lints detect things being declared but not used         | unused-imports, unused-variables, unused-assignments, dead-code, unused-mut, unreachable-code, unreachable-patterns, unused-must-use, unused-unsafe, path-statements, unused-attributes, unused-macros, unused-allocation, unused-doc-comment, unused-extern-crates, unused-features, unused-parens                                                                                                                                                                                    |
| future-incompatible | Lints that detect code that has future-compatibility problems | private-in-public, pub-use-of-private-extern-crate, patterns-in-fns-without-body, safe-extern-statics, invalid-type-param-default, legacy-directory-ownership, legacy-imports, legacy-constructor-visibility, missing-fragment-specifier, illegal-floating-point-literal-pattern, anonymous-parameters, parenthesized-params-in-types-and-modules, late-bound-lifetime-arguments, safe-packed-borrows, tyvar-behind-raw-pointer, unstable-name-collision |

Additionally, there's a `bad-style` lint group that's a deprecated alias for `nonstandard-style`.

Finally, you can also see the table above by invoking `rustc -W help`. This will give you the exact values for the specific
compiler you have installed.
