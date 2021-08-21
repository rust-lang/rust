# The THIR

<!-- toc -->

The THIR ("Typed High-Level Intermediate Representation"), previously called HAIR for
"High-Level Abstract IR", is another IR used by rustc that is generated after
[type checking]. It is (as of <!-- date: 2021-08 --> August 2021) only used for
[MIR construction] and [exhaustiveness checking]. There is also
[an experimental unsafety checker][thir-unsafeck] that operates on the THIR as a replacement for
the current MIR unsafety checker, and can be used instead of the MIR unsafety checker by passing
the `-Z thir-unsafeck` flag to `rustc`.

[type checking]: ./type-checking.md
[MIR construction]: ./mir/construction.md
[exhaustiveness checking]: ./pat-exhaustive-checking.md
[thir-unsafeck]: https://github.com/rust-lang/compiler-team/issues/402

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
[body owners]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir/enum.BodyOwnerKind.html

The THIR lives in [`rustc_mir_build::thir`][thir-docs]. To construct a [`thir::Expr`],
you can use the [`thir_body`] function, passing in the memory arena where the THIR
will be allocated. Dropping this arena will result in the THIR being destroyed,
which is useful to keep peak memory in check. Having a THIR representation of
all bodies of a crate in memory at the same time would be very heavy.

You can get a debug representation of the THIR by passing the `-Zunpretty=thir-tree` flag
to `rustc`. Here is how a function with just the statement `let x = 1 + 2;` gets represented in
THIR:
```rust
Thir {
    // no match arms
    arms: [],
    exprs: [
        // expression 0, a literal with a value of 1
        Expr {
            ty: i32,
            temp_lifetime: Some(Node(6)),
            span: oneplustwo.rs:2:13: 2:14 (#0),
            kind: Literal {
                literal: Const {
                    ty: i32,
                    val: Value(Scalar(0x00000001)),
                },
                user_ty: None,
                const_id: None,
            },
        },
        // expression 1, scope surronding literal 1
        Expr {
            ty: i32,
            temp_lifetime: Some(Node(6)),
            span: oneplustwo.rs:2:13: 2:14 (#0),
            kind: Scope {
                region_scope: Node(1),
                lint_level: Explicit(HirId {
                    owner: DefId(0:3 ~ oneplustwo[6ccc]::main),
                    local_id: 1,
                }),
                // reference to expression 0 above
                value: e0,
            },
        },
        // expression 2, literal 2
        Expr {
            ty: i32,
            temp_lifetime: Some(Node(6)),
            span: oneplustwo.rs:2:17: 2:18 (#0),
            kind: Literal {
                literal: Const {
                    ty: i32,
                    val: Value(Scalar(0x00000002)),
                },
                user_ty: None,
                const_id: None,
            },
        },
        // expression 3, scope surrounding literal 2
        Expr {
            ty: i32,
            temp_lifetime: Some(Node(6)),
            span: oneplustwo.rs:2:17: 2:18 (#0),
            kind: Scope {
                region_scope: Node(2),
                lint_level: Explicit(HirId {
                    owner: DefId(0:3 ~ oneplustwo[6ccc]::main),
                    local_id: 2,
                }),
                // reference to expression 2 above
                value: e2,
            },
        },
        // expression 4, represents 1 + 2
        Expr {
            ty: i32,
            temp_lifetime: Some(Node(6)),
            span: oneplustwo.rs:2:13: 2:18 (#0),
            kind: Binary {
                op: Add,
                // references to scopes surronding literals above
                lhs: e1,
                rhs: e3,
            },
        },
        // expression 5, scope surronding expression 4
        Expr {
            ty: i32,
            temp_lifetime: Some(Node(6)),
            span: oneplustwo.rs:2:13: 2:18 (#0),
            kind: Scope {
                region_scope: Node(3),
                lint_level: Explicit(HirId {
                    owner: DefId(0:3 ~ oneplustwo[6ccc]::main),
                    local_id: 3,
                }),
                value: e4,
            },
        },
        // expression 6, block around statement
        Expr {
            ty: (),
            temp_lifetime: Some(Node(8)),
            span: oneplustwo.rs:1:11: 3:2 (#0),
            kind: Block {
                body: Block {
                    targeted_by_break: false,
                    region_scope: Node(7),
                    opt_destruction_scope: None,
                    span: oneplustwo.rs:1:11: 3:2 (#0),
                    // reference to statement 0 below
                    stmts: [ s0 ],
                    expr: None,
                    safety_mode: Safe,
                },
            },
        },
        // expression 7, scope around block in expression 6
        Expr {
            ty: (),
            temp_lifetime: Some(
                Node(8),
            ),
            span: oneplustwo.rs:1:11: 3:2 (#0),
            kind: Scope {
                region_scope: Node(8),
                lint_level: Explicit(HirId {
                    owner: DefId(0:3 ~ oneplustwo[6ccc]::main),
                    local_id: 8,
                }),
                value: e6,
            },
        },
        // destruction scope around expression 7
        Expr {
            ty: (),
            temp_lifetime: Some(Node(8)),
            span: oneplustwo.rs:1:11: 3:2 (#0),
            kind: Scope {
                region_scope: Destruction(8),
                lint_level: Inherited,
                value: e7,
            },
        },
    ],
    stmts: [
        // let statement
        Stmt {
            kind: Let {
                remainder_scope: Remainder { block: 7, first_statement_index: 0},
                init_scope: Node(6),
                pattern: Pat {
                    ty: i32,
                    span: oneplustwo.rs:2:9: 2:10 (#0),
                    kind: Binding {
                        mutability: Not,
                        name: "x",
                        mode: ByValue,
                        var: HirId {
                            owner: DefId(0:3 ~ oneplustwo[6ccc]::main),
                            local_id: 5,
                        },
                        ty: i32,
                        subpattern: None,
                        is_primary: true,
                    },
                },
                initializer: Some(e5),
                lint_level: Explicit(HirId {
                    owner: DefId(0:3 ~ oneplustwo[6ccc]::main),
                    local_id: 4,
                }),
            },
            opt_destruction_scope: Some(Destruction(6)),
        },
    ],
}
```

[thir-docs]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_mir_build/thir/index.html
[`thir::Expr`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/thir/struct.Expr.html
[`thir_body`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/context/struct.TyCtxt.html#method.thir_body
