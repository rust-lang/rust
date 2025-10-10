# The THIR

The THIR ("Typed High-Level Intermediate Representation"), previously called HAIR for
"High-Level Abstract IR", is another IR used by rustc that is generated after
[type checking]. It is (as of <!-- date-check --> January 2024) used for
[MIR construction], [exhaustiveness checking], and [unsafety checking].

[type checking]: ./type-checking.md
[MIR construction]: ./mir/construction.md
[exhaustiveness checking]: ./pat-exhaustive-checking.md
[unsafety checking]: ./unsafety-checking.md

As the name might suggest, the THIR is a lowered version of the [HIR] where all
the types have been filled in, which is possible after type checking has completed.
But it has some other interesting features that distinguish it from the HIR:

- Like the MIR, the THIR only represents bodies, i.e. "executable code"; this includes
  function bodies, but also `const` initializers, for example. Specifically, all [body owners] have
  THIR created. Consequently, the THIR has no representation for items like `struct`s or `trait`s.

- Each body of THIR is only stored temporarily and is dropped as soon as it's no longer
  needed, as opposed to being stored until the end of the compilation process (which
  is what is done with the HIR).

- Besides making the types of all nodes available, the THIR also has additional
  desugaring compared to the HIR. For example, automatic references and dereferences
  are made explicit, and method calls and overloaded operators are converted into
  plain function calls. Destruction scopes are also made explicit.

- Statements, expressions, and match arms are stored separately. For example, statements in the
  `stmts` array reference expressions by their index (represented as a [`ExprId`]) in the `exprs`
  array.

[HIR]: ./hir.md
[`ExprId`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/thir/struct.ExprId.html
[body owners]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir/hir/enum.BodyOwnerKind.html

The THIR lives in [`rustc_mir_build::thir`][thir-docs]. To construct a [`thir::Expr`],
you can use the [`thir_body`] function, passing in the memory arena where the THIR
will be allocated. Dropping this arena will result in the THIR being destroyed,
which is useful to keep peak memory in check. Having a THIR representation of
all bodies of a crate in memory at the same time would be very heavy.

You can get a debug representation of the THIR by passing the `-Zunpretty=thir-tree` flag
to `rustc`.

To demonstrate, let's use the following example:

```rust
fn main() {
    let x = 1 + 2;
}
```

Here is how that gets represented in THIR (as of <!-- date-check --> Aug 2022):

```rust,no_run
Thir {
    // no match arms
    arms: [],
    exprs: [
        // expression 0, a literal with a value of 1
        Expr {
            ty: i32,
            temp_lifetime: Some(
                Node(1),
            ),
            span: oneplustwo.rs:2:13: 2:14 (#0),
            kind: Literal {
                lit: Spanned {
                    node: Int(
                        1,
                        Unsuffixed,
                    ),
                    span: oneplustwo.rs:2:13: 2:14 (#0),
                },
                neg: false,
            },
        },
        // expression 1, scope surrounding literal 1
        Expr {
            ty: i32,
            temp_lifetime: Some(
                Node(1),
            ),
            span: oneplustwo.rs:2:13: 2:14 (#0),
            kind: Scope {
                // reference to expression 0 above
                region_scope: Node(3),
                lint_level: Explicit(
                    HirId {
                        owner: DefId(0:3 ~ oneplustwo[6932]::main),
                        local_id: 3,
                    },
                ),
                value: e0,
            },
        },
        // expression 2, literal 2
        Expr {
            ty: i32,
            temp_lifetime: Some(
                Node(1),
            ),
            span: oneplustwo.rs:2:17: 2:18 (#0),
            kind: Literal {
                lit: Spanned {
                    node: Int(
                        2,
                        Unsuffixed,
                    ),
                    span: oneplustwo.rs:2:17: 2:18 (#0),
                },
                neg: false,
            },
        },
        // expression 3, scope surrounding literal 2
        Expr {
            ty: i32,
            temp_lifetime: Some(
                Node(1),
            ),
            span: oneplustwo.rs:2:17: 2:18 (#0),
            kind: Scope {
                region_scope: Node(4),
                lint_level: Explicit(
                    HirId {
                        owner: DefId(0:3 ~ oneplustwo[6932]::main),
                        local_id: 4,
                    },
                ),
                // reference to expression 2 above
                value: e2,
            },
        },
        // expression 4, represents 1 + 2
        Expr {
            ty: i32,
            temp_lifetime: Some(
                Node(1),
            ),
            span: oneplustwo.rs:2:13: 2:18 (#0),
            kind: Binary {
                op: Add,
                // references to scopes surrounding literals above
                lhs: e1,
                rhs: e3,
            },
        },
        // expression 5, scope surrounding expression 4
        Expr {
            ty: i32,
            temp_lifetime: Some(
                Node(1),
            ),
            span: oneplustwo.rs:2:13: 2:18 (#0),
            kind: Scope {
                region_scope: Node(5),
                lint_level: Explicit(
                    HirId {
                        owner: DefId(0:3 ~ oneplustwo[6932]::main),
                        local_id: 5,
                    },
                ),
                value: e4,
            },
        },
        // expression 6, block around statement
        Expr {
            ty: (),
            temp_lifetime: Some(
                Node(9),
            ),
            span: oneplustwo.rs:1:11: 3:2 (#0),
            kind: Block {
                body: Block {
                    targeted_by_break: false,
                    region_scope: Node(8),
                    opt_destruction_scope: None,
                    span: oneplustwo.rs:1:11: 3:2 (#0),
                    // reference to statement 0 below
                    stmts: [
                        s0,
                    ],
                    expr: None,
                    safety_mode: Safe,
                },
            },
        },
        // expression 7, scope around block in expression 6
        Expr {
            ty: (),
            temp_lifetime: Some(
                Node(9),
            ),
            span: oneplustwo.rs:1:11: 3:2 (#0),
            kind: Scope {
                region_scope: Node(9),
                lint_level: Explicit(
                    HirId {
                        owner: DefId(0:3 ~ oneplustwo[6932]::main),
                        local_id: 9,
                    },
                ),
                value: e6,
            },
        },
        // destruction scope around expression 7
        Expr {
            ty: (),
            temp_lifetime: Some(
                Node(9),
            ),
            span: oneplustwo.rs:1:11: 3:2 (#0),
            kind: Scope {
                region_scope: Destruction(9),
                lint_level: Inherited,
                value: e7,
            },
        },
    ],
    stmts: [
        // let statement
        Stmt {
            kind: Let {
                remainder_scope: Remainder { block: 8, first_statement_index: 0},
                init_scope: Node(1),
                pattern: Pat {
                    ty: i32,
                    span: oneplustwo.rs:2:9: 2:10 (#0),
                    kind: Binding {
                        mutability: Not,
                        name: "x",
                        mode: ByValue,
                        var: LocalVarId(
                            HirId {
                                owner: DefId(0:3 ~ oneplustwo[6932]::main),
                                local_id: 7,
                            },
                        ),
                        ty: i32,
                        subpattern: None,
                        is_primary: true,
                    },
                },
                initializer: Some(
                    e5,
                ),
                else_block: None,
                lint_level: Explicit(
                    HirId {
                        owner: DefId(0:3 ~ oneplustwo[6932]::main),
                        local_id: 6,
                    },
                ),
            },
            opt_destruction_scope: Some(
                Destruction(1),
            ),
        },
    ],
}
```

[thir-docs]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_mir_build/thir/index.html
[`thir::Expr`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/thir/struct.Expr.html
[`thir_body`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/context/struct.TyCtxt.html#method.thir_body
