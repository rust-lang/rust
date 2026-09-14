# The THIR

The THIR ("Typed High-Level Intermediate Representation"), previously called HAIR for
"High-Level Abstract IR", is another IR used by rustc that is generated after
[type checking]. It is (as of <!-- date-check --> January 2024) used for
[MIR construction], [exhaustiveness checking], and [unsafety checking].

[type checking]: ./hir-typeck/summary.md
[MIR construction]: ./mir/construction.md
[exhaustiveness checking]: ./pat-exhaustive-checking.md
[unsafety checking]: ./unsafety-checking.md

As the name might suggest, the THIR is a lowered version of the [HIR] where all
the types have been filled in, which is possible after type checking has completed.
But it has some other interesting features that distinguish it from the HIR:

- Like the MIR, the THIR only represents bodies, i.e. "executable code"; this includes
  function bodies, but also `const` initializers, for example.
  Specifically, all [body owners] have THIR created.
  Consequently, the THIR has no representation for items like `struct`s or `trait`s.

- Each body of THIR is only stored temporarily and is dropped as soon as it's no longer
  needed, as opposed to being stored until the end of the compilation process (which
  is what is done with the HIR).

- Besides making the types of all nodes available, the THIR also has additional
  desugaring compared to the HIR.
  For example, automatic references and dereferences
  are made explicit, and method calls and overloaded operators are converted into
  plain function calls.
  Destruction scopes are also made explicit.

- Statements, expressions, match arms, blocks, and parameters are stored separately.
  For example,
  statements in the `stmts` array reference expressions by their index (represented as a
  [`ExprId`]) in the `exprs` array.

[HIR]: ./hir.md
[`ExprId`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/thir/struct.ExprId.html
[body owners]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir/hir/enum.BodyOwnerKind.html

The THIR lives in [`rustc_mir_build::thir`][thir-docs].
To construct a [`thir::Expr`],
you can use the [`thir_body`] function, passing in the memory arena where the THIR
will be allocated.
Dropping this arena will result in the THIR being destroyed,
which is useful to keep peak memory in check.
Having a THIR representation of
all bodies of a crate in memory at the same time would be very heavy.

You can get a debug representation of the THIR by passing the `-Zunpretty=thir-flat` flag
to `rustc`.

To demonstrate, let's use the following example:

```rust
fn main() {
    let x = 1 + 2;
}
```

Here is how that gets represented in THIR (as of <!-- date-check --> Jul 2026):

```rust,no_run
DefId(0:3 ~ main[26fd]::main):
Thir {
    body_type: Fn(
        fn(),
    ),
    attributes: {},
    // no match arms
    arms: [],
    blocks: [
        Block {
            targeted_by_break: false,
            region_scope: Node(1),
            span: main.rs:1:11: 3:2 (#0),
            stmts: [
                s0,
            ],
            expr: None,
            safety_mode: Safe,
        },
    ],
    exprs: [
        // expression 0, a literal with a value of 1
        Expr {
            kind: Literal {
                lit: Spanned {
                    node: Int(
                        Pu128(
                            1,
                        ),
                        Unsuffixed,
                    ),
                    span: main.rs:2:13: 2:14 (#0),
                },
                neg: false,
            },
            ty: i32,
            temp_scope_id: 4,
            span: main.rs:2:13: 2:14 (#0),
        },
        // expression 1, scope surrounding literal 1
        Expr {
            kind: Scope {
                region_scope: Node(4),
                hir_id: HirId(DefId(0:3 ~ main[26fd]::main).4),
                // reference to expression 0 above
                value: e0,
            },
            ty: i32,
            temp_scope_id: 4,
            span: main.rs:2:13: 2:14 (#0),
        },
        // expression 2, literal 2
        Expr {
            kind: Literal {
                lit: Spanned {
                    node: Int(
                        Pu128(
                            2,
                        ),
                        Unsuffixed,
                    ),
                    span: main.rs:2:17: 2:18 (#0),
                },
                neg: false,
            },
            ty: i32,
            temp_scope_id: 5,
            span: main.rs:2:17: 2:18 (#0),
        },
        // expression 3, scope surrounding literal 2
        Expr {
            kind: Scope {
                region_scope: Node(5),
                hir_id: HirId(DefId(0:3 ~ main[26fd]::main).5),
                // reference to expression 0 above
                value: e2,
            },
            ty: i32,
            temp_scope_id: 5,
            span: main.rs:2:17: 2:18 (#0),
        },
        // expression 4, represents 1 + 2
        Expr {
            kind: Binary {
                op: Add,
                // references to scopes surrounding literals above
                lhs: e1,
                rhs: e3,
            },
            ty: i32,
            temp_scope_id: 3,
            span: main.rs:2:13: 2:18 (#0),
        },
        // expression 5, scope surrounding expression 4
        Expr {
            kind: Scope {
                region_scope: Node(3),
                hir_id: HirId(DefId(0:3 ~ main[26fd]::main).3),
                value: e4,
            },
            ty: i32,
            temp_scope_id: 3,
            span: main.rs:2:13: 2:18 (#0),
        },
        // expression 6, block around statement
        Expr {
            kind: Block {
                block: b0,
            },
            ty: (),
            temp_scope_id: 8,
            span: main.rs:1:11: 3:2 (#0),
        },
        // expression 7, scope around block in expression 6
        Expr {
            kind: Scope {
                region_scope: Node(8),
                hir_id: HirId(DefId(0:3 ~ main[26fd]::main).8),
                value: e6,
            },
            ty: (),
            temp_scope_id: 8,
            span: main.rs:1:11: 3:2 (#0),
        },
    ],
    stmts: [
        Stmt {
            kind: Let {
                remainder_scope: Remainder { block: 1, first_statement_index: 0},
                init_scope: Node(2),
                pattern: Pat {
                    ty: i32,
                    span: main.rs:2:9: 2:10 (#0),
                    extra: None,
                    kind: Binding {
                        name: "x",
                        mode: BindingMode(
                            No,
                            Not,
                        ),
                        var: LocalVarId(
                            HirId(DefId(0:3 ~ main[26fd]::main).7),
                        ),
                        ty: i32,
                        subpattern: None,
                        is_primary: true,
                        is_shorthand: false,
                    },
                },
                initializer: Some(
                    e5,
                ),
                else_block: None,
                hir_id: HirId(DefId(0:3 ~ main[26fd]::main).6),
                span: main.rs:2:5: 2:18 (#0),
            },
        },
    ],
    params: [],
}
```

[thir-docs]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_mir_build/thir/index.html
[`thir::Expr`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/thir/struct.Expr.html
[`thir_body`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/context/struct.TyCtxt.html#method.thir_body
