# AST validation

_AST validation_ is a separate AST pass that visits each
item in the tree and performs simple checks. This pass
doesn't perform any complex analysis, type checking or
name resolution.

Before performing any validation, the compiler first expands
the macros. Then this pass performs validations to check
that each AST item is in the correct state. And when this pass
is done, the compiler runs the crate resolution pass.

## Validations

Validations are defined in `AstValidator` type, which 
itself is located in `rustc_ast_passes` crate. This
type implements various simple checks which emit errors
when certain language rules are broken.

In addition, `AstValidator` implements `Visitor` trait
that defines how to visit AST items (which can be functions,
traits, enums, etc).

For each item, visitor performs specific checks. For
example, when visiting a function declaration,
`AstValidator` checks that the function has:

* no more than `u16::MAX` parameters;
* c-variadic argument goes the last in the declaration;
* documentation comments aren't applied to function parameters;
* and other validations.
