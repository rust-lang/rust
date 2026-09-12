# The MIR type-check

<!-- TODO: Move this to the analysis folder. -->

"Canonical" type checking (ignoring lifetimes) for Rust happens in the HIR.
Despite this, we also do a [type checking pass in MIR][type_check].

The MIR is our fully-typed intermediate representation, the types of all items and the contents of their bodies are known by this point and by constructing the MIR we know its types are correct.
The reason to do a type checking pass on this already-typed, already-checked IR is to accumulate information about lifetimes[^lifetimes] for borrow checking.
See: [`borrowck_collect_region_constraints`][borrowck_collect_region_constraints].

[^lifetimes]: AKA regions AKA loans.

Doing this additional type checking pass on it also allows us to check our working:
If something fails in MIR type checking that passed in HIR type checking, something has gone wrong.

Maintaining "MIR type checking should succeed if HIR type checking succeeds" is nontrivial.
One major reason for this is that type checking MIR involves erasing the existing lifetimes and replacing them with new unconstrained lifetime variables, while in HIR lifetimes get inferred but not checked.
In this way, HIR type checking and MIR type checking each work with subtly different information.

The erase-and-re-infer strategy in MIR is called [Region Uniquification](#region-uniquification).

## Region Uniquification

TODO: Talk more about Region Uniquification and any still-existing use of it.

[borrowck_collect_region_constraints]: https://doc.rust-lang.org/stable/nightly-rustc/rustc_borrowck/fn.borrowck_collect_region_constraints.html
[type_check]: https://doc.rust-lang.org/stable/nightly-rustc/rustc_borrowck/type_check/fn.type_check.html

## User types

At the start of MIR type checking, we replace all regions in the body with new unconstrained regions.
However, this would cause us to accept the following program:
```rust
fn foo<'a>(x: &'a u32) {
    let y: &'static u32 = x;
}
```
By erasing the lifetimes in the type of `y` we no longer know that it is supposed to be `'static`,
ignoring the intentions of the user.

To deal with this we remember all places where the user explicitly mentioned a type during
HIR type-check as [`CanonicalUserTypeAnnotations`][annot].

There are two different annotations we care about:
- explicit type ascriptions, e.g. `let y: &'static u32` results in `UserType::Ty(&'static u32)`.
- explicit generic arguments, e.g. `x.foo<&'a u32, Vec<String>>`
results in `UserType::TypeOf(foo_def_id, [&'a u32, Vec<String>])`.

As we do not want the region inference from the HIR type-check to influence MIR typeck,
we store the user type right after lowering it from the HIR.
This means that it may still contain inference variables,
which is why we are using **canonical** user type annotations.
We replace all inference variables with existential bound variables instead.
Something like `let x: Vec<_>` would therefore result in `exists<T> UserType::Ty(Vec<T>)`.

A pattern like `let Foo(x): Foo<&'a u32>` has a user type `Foo<&'a u32>` but
the actual type of `x` should only be `&'a u32`.
For this, we use a [`UserTypeProjection`][proj].

In the MIR, we deal with user types in two slightly different ways.

Given a MIR local corresponding to a variable in a pattern which has an explicit type annotation,
we require the type of that local to be equal to the type of the [`UserTypeProjection`][proj].
This is directly stored in the [`LocalDecl`][decl].

We also constrain the type of scrutinee expressions, e.g. the type of `x` in `let _: &'a u32 = x;`.
Here `T_x` only has to be a subtype of the user type, so we instead use
[`StatementKind::AscribeUserType`][stmt] for that.

Note that we do not directly use the user type as the MIR typechecker
doesn't really deal with type and const inference variables.
We instead store the final [`inferred_type`][inf] from the HIR type-checker.
During MIR typeck, we then replace its regions with new nll inference vars
and relate it with the actual `UserType` to get the correct region constraints again.

After the MIR type-check, all user type annotations get discarded as they aren't needed anymore.

[annot]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.CanonicalUserTypeAnnotation.html
[proj]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/mir/struct.UserTypeProjection.html
[decl]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/mir/struct.LocalDecl.html
[stmt]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/mir/enum.StatementKind.html#variant.AscribeUserType
[inf]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.CanonicalUserTypeAnnotation.html#structfield.inferred_ty
