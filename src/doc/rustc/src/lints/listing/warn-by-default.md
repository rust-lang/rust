# Warn-by-default lints

These lints are all set to the 'warn' level by default.

                                    const-err  warn     constant evaluation detected erroneous expression
                                    dead-code  warn     detect unused, unexported items
                                   deprecated  warn     detects use of deprecated items
       illegal-floating-point-literal-pattern  warn     floating-point literals cannot be used in patterns
                              improper-ctypes  warn     proper use of libc types in foreign modules
                 incoherent-fundamental-impls  warn     potentially-conflicting impls were erroneously allowed
                late-bound-lifetime-arguments  warn     detects generic lifetime arguments in path segments with late bound lifetime parameters
                         non-camel-case-types  warn     types, variants, traits and type parameters should have camel case names
                 non-shorthand-field-patterns  warn     using `Struct { x: x }` instead of `Struct { x }` in a pattern
                               non-snake-case  warn     variables, methods, functions, lifetime parameters and modules should have snake case names
                       non-upper-case-globals  warn     static constants should have uppercase identifiers
                      no-mangle-generic-items  warn     generic items must be mangled
                         overflowing-literals  warn     literal out of range for its type
                              path-statements  warn     path statements with no effect
                 patterns-in-fns-without-body  warn     patterns in functions without body were erroneously allowed
                            plugin-as-library  warn     compiler plugin used as ordinary library in non-plugin crate
                            private-in-public  warn     detect private items in public interfaces not caught by the old implementation
                        private-no-mangle-fns  warn     functions marked #[no_mangle] should be exported
                    private-no-mangle-statics  warn     statics marked #[no_mangle] should be exported
                    renamed-and-removed-lints  warn     lints that have been renamed or removed
                          safe-packed-borrows  warn     safe borrows of fields of packed structs were was erroneously allowed
                              stable-features  warn     stable features found in #[feature] directive
                            type-alias-bounds  warn     bounds in type aliases are not enforced
                     tyvar-behind-raw-pointer  warn     raw pointer to an inference variable
                      unconditional-recursion  warn     functions that cannot return without calling themselves
                      unions-with-drop-fields  warn     use of unions that contain fields with possibly non-trivial drop code
                                unknown-lints  warn     unrecognized lint attribute
                             unreachable-code  warn     detects unreachable code paths
                         unreachable-patterns  warn     detects unreachable patterns
                      unstable-name-collision  warn     detects name collision with an existing but unstable method
                            unused-allocation  warn     detects unnecessary allocations that can be eliminated
                           unused-assignments  warn     detect assignments that will never be read
                            unused-attributes  warn     detects attributes that were not used by the compiler
                           unused-comparisons  warn     comparisons made useless by limits of the types involved
                           unused-doc-comment  warn     detects doc comments that aren't used by rustdoc
                              unused-features  warn     unused or unknown features found in crate-level #[feature] directives
                               unused-imports  warn     imports that are never used
                                unused-macros  warn     detects macros that were not used
                              unused-must-use  warn     unused result of a type flagged as #[must_use]
                                   unused-mut  warn     detect mut variables which don't need to be mutable
                                unused-parens  warn     `if`, `match`, `while` and `return` do not need parentheses
                                unused-unsafe  warn     unnecessary use of an `unsafe` block
                             unused-variables  warn     detect variables which are not used in any way
                                     warnings  warn     mass-change the level for lints which produce warnings
                                   while-true  warn     suggest using `loop { }` instead of `while true { }`