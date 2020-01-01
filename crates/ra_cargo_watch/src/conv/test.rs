//! This module contains the large and verbose snapshot tests for the
//! conversions between `cargo check` json and LSP diagnostics.
use crate::*;

fn parse_diagnostic(val: &str) -> cargo_metadata::diagnostic::Diagnostic {
    serde_json::from_str::<cargo_metadata::diagnostic::Diagnostic>(val).unwrap()
}

#[test]
#[cfg(not(windows))]
fn snap_rustc_incompatible_type_for_trait() {
    let diag = parse_diagnostic(
        r##"{
            "message": "method `next` has an incompatible type for trait",
            "code": {
                "code": "E0053",
                "explanation": "\nThe parameters of any trait method must match between a trait implementation\nand the trait definition.\n\nHere are a couple examples of this error:\n\n```compile_fail,E0053\ntrait Foo {\n    fn foo(x: u16);\n    fn bar(&self);\n}\n\nstruct Bar;\n\nimpl Foo for Bar {\n    // error, expected u16, found i16\n    fn foo(x: i16) { }\n\n    // error, types differ in mutability\n    fn bar(&mut self) { }\n}\n```\n"
            },
            "level": "error",
            "spans": [
                {
                    "file_name": "compiler/ty/list_iter.rs",
                    "byte_start": 1307,
                    "byte_end": 1350,
                    "line_start": 52,
                    "line_end": 52,
                    "column_start": 5,
                    "column_end": 48,
                    "is_primary": true,
                    "text": [
                        {
                            "text": "    fn next(&self) -> Option<&'list ty::Ref<M>> {",
                            "highlight_start": 5,
                            "highlight_end": 48
                        }
                    ],
                    "label": "types differ in mutability",
                    "suggested_replacement": null,
                    "suggestion_applicability": null,
                    "expansion": null
                }
            ],
            "children": [
                {
                    "message": "expected type `fn(&mut ty::list_iter::ListIterator<'list, M>) -> std::option::Option<&ty::Ref<M>>`\n   found type `fn(&ty::list_iter::ListIterator<'list, M>) -> std::option::Option<&'list ty::Ref<M>>`",
                    "code": null,
                    "level": "note",
                    "spans": [],
                    "children": [],
                    "rendered": null
                }
            ],
            "rendered": "error[E0053]: method `next` has an incompatible type for trait\n  --> compiler/ty/list_iter.rs:52:5\n   |\n52 |     fn next(&self) -> Option<&'list ty::Ref<M>> {\n   |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ types differ in mutability\n   |\n   = note: expected type `fn(&mut ty::list_iter::ListIterator<'list, M>) -> std::option::Option<&ty::Ref<M>>`\n              found type `fn(&ty::list_iter::ListIterator<'list, M>) -> std::option::Option<&'list ty::Ref<M>>`\n\n"
        }
        "##,
    );

    let workspace_root = PathBuf::from("/test/");
    let diag = map_rust_diagnostic_to_lsp(&diag, &workspace_root).expect("couldn't map diagnostic");
    insta::assert_debug_snapshot!(diag);
}

#[test]
#[cfg(not(windows))]
fn snap_rustc_unused_variable() {
    let diag = parse_diagnostic(
        r##"{
"message": "unused variable: `foo`",
"code": {
    "code": "unused_variables",
    "explanation": null
},
"level": "warning",
"spans": [
    {
        "file_name": "driver/subcommand/repl.rs",
        "byte_start": 9228,
        "byte_end": 9231,
        "line_start": 291,
        "line_end": 291,
        "column_start": 9,
        "column_end": 12,
        "is_primary": true,
        "text": [
            {
                "text": "    let foo = 42;",
                "highlight_start": 9,
                "highlight_end": 12
            }
        ],
        "label": null,
        "suggested_replacement": null,
        "suggestion_applicability": null,
        "expansion": null
    }
],
"children": [
    {
        "message": "#[warn(unused_variables)] on by default",
        "code": null,
        "level": "note",
        "spans": [],
        "children": [],
        "rendered": null
    },
    {
        "message": "consider prefixing with an underscore",
        "code": null,
        "level": "help",
        "spans": [
            {
                "file_name": "driver/subcommand/repl.rs",
                "byte_start": 9228,
                "byte_end": 9231,
                "line_start": 291,
                "line_end": 291,
                "column_start": 9,
                "column_end": 12,
                "is_primary": true,
                "text": [
                    {
                        "text": "    let foo = 42;",
                        "highlight_start": 9,
                        "highlight_end": 12
                    }
                ],
                "label": null,
                "suggested_replacement": "_foo",
                "suggestion_applicability": "MachineApplicable",
                "expansion": null
            }
        ],
        "children": [],
        "rendered": null
    }
],
"rendered": "warning: unused variable: `foo`\n   --> driver/subcommand/repl.rs:291:9\n    |\n291 |     let foo = 42;\n    |         ^^^ help: consider prefixing with an underscore: `_foo`\n    |\n    = note: #[warn(unused_variables)] on by default\n\n"
}"##,
    );

    let workspace_root = PathBuf::from("/test/");
    let diag = map_rust_diagnostic_to_lsp(&diag, &workspace_root).expect("couldn't map diagnostic");
    insta::assert_debug_snapshot!(diag);
}

#[test]
#[cfg(not(windows))]
fn snap_rustc_wrong_number_of_parameters() {
    let diag = parse_diagnostic(
        r##"{
"message": "this function takes 2 parameters but 3 parameters were supplied",
"code": {
    "code": "E0061",
    "explanation": "\nThe number of arguments passed to a function must match the number of arguments\nspecified in the function signature.\n\nFor example, a function like:\n\n```\nfn f(a: u16, b: &str) {}\n```\n\nMust always be called with exactly two arguments, e.g., `f(2, \"test\")`.\n\nNote that Rust does not have a notion of optional function arguments or\nvariadic functions (except for its C-FFI).\n"
},
"level": "error",
"spans": [
    {
        "file_name": "compiler/ty/select.rs",
        "byte_start": 8787,
        "byte_end": 9241,
        "line_start": 219,
        "line_end": 231,
        "column_start": 5,
        "column_end": 6,
        "is_primary": false,
        "text": [
            {
                "text": "    pub fn add_evidence(",
                "highlight_start": 5,
                "highlight_end": 25
            },
            {
                "text": "        &mut self,",
                "highlight_start": 1,
                "highlight_end": 19
            },
            {
                "text": "        target_poly: &ty::Ref<ty::Poly>,",
                "highlight_start": 1,
                "highlight_end": 41
            },
            {
                "text": "        evidence_poly: &ty::Ref<ty::Poly>,",
                "highlight_start": 1,
                "highlight_end": 43
            },
            {
                "text": "    ) {",
                "highlight_start": 1,
                "highlight_end": 8
            },
            {
                "text": "        match target_poly {",
                "highlight_start": 1,
                "highlight_end": 28
            },
            {
                "text": "            ty::Ref::Var(tvar, _) => self.add_var_evidence(tvar, evidence_poly),",
                "highlight_start": 1,
                "highlight_end": 81
            },
            {
                "text": "            ty::Ref::Fixed(target_ty) => {",
                "highlight_start": 1,
                "highlight_end": 43
            },
            {
                "text": "                let evidence_ty = evidence_poly.resolve_to_ty();",
                "highlight_start": 1,
                "highlight_end": 65
            },
            {
                "text": "                self.add_evidence_ty(target_ty, evidence_poly, evidence_ty)",
                "highlight_start": 1,
                "highlight_end": 76
            },
            {
                "text": "            }",
                "highlight_start": 1,
                "highlight_end": 14
            },
            {
                "text": "        }",
                "highlight_start": 1,
                "highlight_end": 10
            },
            {
                "text": "    }",
                "highlight_start": 1,
                "highlight_end": 6
            }
        ],
        "label": "defined here",
        "suggested_replacement": null,
        "suggestion_applicability": null,
        "expansion": null
    },
    {
        "file_name": "compiler/ty/select.rs",
        "byte_start": 4045,
        "byte_end": 4057,
        "line_start": 104,
        "line_end": 104,
        "column_start": 18,
        "column_end": 30,
        "is_primary": true,
        "text": [
            {
                "text": "            self.add_evidence(target_fixed, evidence_fixed, false);",
                "highlight_start": 18,
                "highlight_end": 30
            }
        ],
        "label": "expected 2 parameters",
        "suggested_replacement": null,
        "suggestion_applicability": null,
        "expansion": null
    }
],
"children": [],
"rendered": "error[E0061]: this function takes 2 parameters but 3 parameters were supplied\n   --> compiler/ty/select.rs:104:18\n    |\n104 |               self.add_evidence(target_fixed, evidence_fixed, false);\n    |                    ^^^^^^^^^^^^ expected 2 parameters\n...\n219 | /     pub fn add_evidence(\n220 | |         &mut self,\n221 | |         target_poly: &ty::Ref<ty::Poly>,\n222 | |         evidence_poly: &ty::Ref<ty::Poly>,\n...   |\n230 | |         }\n231 | |     }\n    | |_____- defined here\n\n"
}"##,
    );

    let workspace_root = PathBuf::from("/test/");
    let diag = map_rust_diagnostic_to_lsp(&diag, &workspace_root).expect("couldn't map diagnostic");
    insta::assert_debug_snapshot!(diag);
}

#[test]
#[cfg(not(windows))]
fn snap_clippy_pass_by_ref() {
    let diag = parse_diagnostic(
        r##"{
"message": "this argument is passed by reference, but would be more efficient if passed by value",
"code": {
    "code": "clippy::trivially_copy_pass_by_ref",
    "explanation": null
},
"level": "warning",
"spans": [
    {
        "file_name": "compiler/mir/tagset.rs",
        "byte_start": 941,
        "byte_end": 946,
        "line_start": 42,
        "line_end": 42,
        "column_start": 24,
        "column_end": 29,
        "is_primary": true,
        "text": [
            {
                "text": "    pub fn is_disjoint(&self, other: Self) -> bool {",
                "highlight_start": 24,
                "highlight_end": 29
            }
        ],
        "label": null,
        "suggested_replacement": null,
        "suggestion_applicability": null,
        "expansion": null
    }
],
"children": [
    {
        "message": "lint level defined here",
        "code": null,
        "level": "note",
        "spans": [
            {
                "file_name": "compiler/lib.rs",
                "byte_start": 8,
                "byte_end": 19,
                "line_start": 1,
                "line_end": 1,
                "column_start": 9,
                "column_end": 20,
                "is_primary": true,
                "text": [
                    {
                        "text": "#![warn(clippy::all)]",
                        "highlight_start": 9,
                        "highlight_end": 20
                    }
                ],
                "label": null,
                "suggested_replacement": null,
                "suggestion_applicability": null,
                "expansion": null
            }
        ],
        "children": [],
        "rendered": null
    },
    {
        "message": "#[warn(clippy::trivially_copy_pass_by_ref)] implied by #[warn(clippy::all)]",
        "code": null,
        "level": "note",
        "spans": [],
        "children": [],
        "rendered": null
    },
    {
        "message": "for further information visit https://rust-lang.github.io/rust-clippy/master/index.html#trivially_copy_pass_by_ref",
        "code": null,
        "level": "help",
        "spans": [],
        "children": [],
        "rendered": null
    },
    {
        "message": "consider passing by value instead",
        "code": null,
        "level": "help",
        "spans": [
            {
                "file_name": "compiler/mir/tagset.rs",
                "byte_start": 941,
                "byte_end": 946,
                "line_start": 42,
                "line_end": 42,
                "column_start": 24,
                "column_end": 29,
                "is_primary": true,
                "text": [
                    {
                        "text": "    pub fn is_disjoint(&self, other: Self) -> bool {",
                        "highlight_start": 24,
                        "highlight_end": 29
                    }
                ],
                "label": null,
                "suggested_replacement": "self",
                "suggestion_applicability": "Unspecified",
                "expansion": null
            }
        ],
        "children": [],
        "rendered": null
    }
],
"rendered": "warning: this argument is passed by reference, but would be more efficient if passed by value\n  --> compiler/mir/tagset.rs:42:24\n   |\n42 |     pub fn is_disjoint(&self, other: Self) -> bool {\n   |                        ^^^^^ help: consider passing by value instead: `self`\n   |\nnote: lint level defined here\n  --> compiler/lib.rs:1:9\n   |\n1  | #![warn(clippy::all)]\n   |         ^^^^^^^^^^^\n   = note: #[warn(clippy::trivially_copy_pass_by_ref)] implied by #[warn(clippy::all)]\n   = help: for further information visit https://rust-lang.github.io/rust-clippy/master/index.html#trivially_copy_pass_by_ref\n\n"
}"##,
    );

    let workspace_root = PathBuf::from("/test/");
    let diag = map_rust_diagnostic_to_lsp(&diag, &workspace_root).expect("couldn't map diagnostic");
    insta::assert_debug_snapshot!(diag);
}

#[test]
#[cfg(not(windows))]
fn snap_rustc_mismatched_type() {
    let diag = parse_diagnostic(
        r##"{
"message": "mismatched types",
"code": {
    "code": "E0308",
    "explanation": "\nThis error occurs when the compiler was unable to infer the concrete type of a\nvariable. It can occur for several cases, the most common of which is a\nmismatch in the expected type that the compiler inferred for a variable's\ninitializing expression, and the actual type explicitly assigned to the\nvariable.\n\nFor example:\n\n```compile_fail,E0308\nlet x: i32 = \"I am not a number!\";\n//     ~~~   ~~~~~~~~~~~~~~~~~~~~\n//      |             |\n//      |    initializing expression;\n//      |    compiler infers type `&str`\n//      |\n//    type `i32` assigned to variable `x`\n```\n"
},
"level": "error",
"spans": [
    {
        "file_name": "runtime/compiler_support.rs",
        "byte_start": 1589,
        "byte_end": 1594,
        "line_start": 48,
        "line_end": 48,
        "column_start": 65,
        "column_end": 70,
        "is_primary": true,
        "text": [
            {
                "text": "    let layout = alloc::Layout::from_size_align_unchecked(size, align);",
                "highlight_start": 65,
                "highlight_end": 70
            }
        ],
        "label": "expected usize, found u32",
        "suggested_replacement": null,
        "suggestion_applicability": null,
        "expansion": null
    }
],
"children": [],
"rendered": "error[E0308]: mismatched types\n  --> runtime/compiler_support.rs:48:65\n   |\n48 |     let layout = alloc::Layout::from_size_align_unchecked(size, align);\n   |                                                                 ^^^^^ expected usize, found u32\n\n"
}"##,
    );

    let workspace_root = PathBuf::from("/test/");
    let diag = map_rust_diagnostic_to_lsp(&diag, &workspace_root).expect("couldn't map diagnostic");
    insta::assert_debug_snapshot!(diag);
}

#[test]
#[cfg(not(windows))]
fn snap_handles_macro_location() {
    let diag = parse_diagnostic(
        r##"{
"rendered": "error[E0277]: can't compare `{integer}` with `&str`\n --> src/main.rs:2:5\n  |\n2 |     assert_eq!(1, \"love\");\n  |     ^^^^^^^^^^^^^^^^^^^^^^ no implementation for `{integer} == &str`\n  |\n  = help: the trait `std::cmp::PartialEq<&str>` is not implemented for `{integer}`\n  = note: this error originates in a macro outside of the current crate (in Nightly builds, run with -Z external-macro-backtrace for more info)\n\n",
"children": [
    {
        "children": [],
        "code": null,
        "level": "help",
        "message": "the trait `std::cmp::PartialEq<&str>` is not implemented for `{integer}`",
        "rendered": null,
        "spans": []
    }
],
"code": {
    "code": "E0277",
    "explanation": "\nYou tried to use a type which doesn't implement some trait in a place which\nexpected that trait. Erroneous code example:\n\n```compile_fail,E0277\n// here we declare the Foo trait with a bar method\ntrait Foo {\n    fn bar(&self);\n}\n\n// we now declare a function which takes an object implementing the Foo trait\nfn some_func<T: Foo>(foo: T) {\n    foo.bar();\n}\n\nfn main() {\n    // we now call the method with the i32 type, which doesn't implement\n    // the Foo trait\n    some_func(5i32); // error: the trait bound `i32 : Foo` is not satisfied\n}\n```\n\nIn order to fix this error, verify that the type you're using does implement\nthe trait. Example:\n\n```\ntrait Foo {\n    fn bar(&self);\n}\n\nfn some_func<T: Foo>(foo: T) {\n    foo.bar(); // we can now use this method since i32 implements the\n               // Foo trait\n}\n\n// we implement the trait on the i32 type\nimpl Foo for i32 {\n    fn bar(&self) {}\n}\n\nfn main() {\n    some_func(5i32); // ok!\n}\n```\n\nOr in a generic context, an erroneous code example would look like:\n\n```compile_fail,E0277\nfn some_func<T>(foo: T) {\n    println!(\"{:?}\", foo); // error: the trait `core::fmt::Debug` is not\n                           //        implemented for the type `T`\n}\n\nfn main() {\n    // We now call the method with the i32 type,\n    // which *does* implement the Debug trait.\n    some_func(5i32);\n}\n```\n\nNote that the error here is in the definition of the generic function: Although\nwe only call it with a parameter that does implement `Debug`, the compiler\nstill rejects the function: It must work with all possible input types. In\norder to make this example compile, we need to restrict the generic type we're\naccepting:\n\n```\nuse std::fmt;\n\n// Restrict the input type to types that implement Debug.\nfn some_func<T: fmt::Debug>(foo: T) {\n    println!(\"{:?}\", foo);\n}\n\nfn main() {\n    // Calling the method is still fine, as i32 implements Debug.\n    some_func(5i32);\n\n    // This would fail to compile now:\n    // struct WithoutDebug;\n    // some_func(WithoutDebug);\n}\n```\n\nRust only looks at the signature of the called function, as such it must\nalready specify all requirements that will be used for every type parameter.\n"
},
"level": "error",
"message": "can't compare `{integer}` with `&str`",
"spans": [
    {
        "byte_end": 155,
        "byte_start": 153,
        "column_end": 33,
        "column_start": 31,
        "expansion": {
            "def_site_span": {
                "byte_end": 940,
                "byte_start": 0,
                "column_end": 6,
                "column_start": 1,
                "expansion": null,
                "file_name": "<::core::macros::assert_eq macros>",
                "is_primary": false,
                "label": null,
                "line_end": 36,
                "line_start": 1,
                "suggested_replacement": null,
                "suggestion_applicability": null,
                "text": [
                    {
                        "highlight_end": 35,
                        "highlight_start": 1,
                        "text": "($ left : expr, $ right : expr) =>"
                    },
                    {
                        "highlight_end": 3,
                        "highlight_start": 1,
                        "text": "({"
                    },
                    {
                        "highlight_end": 33,
                        "highlight_start": 1,
                        "text": "     match (& $ left, & $ right)"
                    },
                    {
                        "highlight_end": 7,
                        "highlight_start": 1,
                        "text": "     {"
                    },
                    {
                        "highlight_end": 34,
                        "highlight_start": 1,
                        "text": "         (left_val, right_val) =>"
                    },
                    {
                        "highlight_end": 11,
                        "highlight_start": 1,
                        "text": "         {"
                    },
                    {
                        "highlight_end": 46,
                        "highlight_start": 1,
                        "text": "             if ! (* left_val == * right_val)"
                    },
                    {
                        "highlight_end": 15,
                        "highlight_start": 1,
                        "text": "             {"
                    },
                    {
                        "highlight_end": 25,
                        "highlight_start": 1,
                        "text": "                 panic !"
                    },
                    {
                        "highlight_end": 57,
                        "highlight_start": 1,
                        "text": "                 (r#\"assertion failed: `(left == right)`"
                    },
                    {
                        "highlight_end": 16,
                        "highlight_start": 1,
                        "text": "  left: `{:?}`,"
                    },
                    {
                        "highlight_end": 18,
                        "highlight_start": 1,
                        "text": " right: `{:?}`\"#,"
                    },
                    {
                        "highlight_end": 47,
                        "highlight_start": 1,
                        "text": "                  & * left_val, & * right_val)"
                    },
                    {
                        "highlight_end": 15,
                        "highlight_start": 1,
                        "text": "             }"
                    },
                    {
                        "highlight_end": 11,
                        "highlight_start": 1,
                        "text": "         }"
                    },
                    {
                        "highlight_end": 7,
                        "highlight_start": 1,
                        "text": "     }"
                    },
                    {
                        "highlight_end": 42,
                        "highlight_start": 1,
                        "text": " }) ; ($ left : expr, $ right : expr,) =>"
                    },
                    {
                        "highlight_end": 49,
                        "highlight_start": 1,
                        "text": "({ $ crate :: assert_eq ! ($ left, $ right) }) ;"
                    },
                    {
                        "highlight_end": 53,
                        "highlight_start": 1,
                        "text": "($ left : expr, $ right : expr, $ ($ arg : tt) +) =>"
                    },
                    {
                        "highlight_end": 3,
                        "highlight_start": 1,
                        "text": "({"
                    },
                    {
                        "highlight_end": 37,
                        "highlight_start": 1,
                        "text": "     match (& ($ left), & ($ right))"
                    },
                    {
                        "highlight_end": 7,
                        "highlight_start": 1,
                        "text": "     {"
                    },
                    {
                        "highlight_end": 34,
                        "highlight_start": 1,
                        "text": "         (left_val, right_val) =>"
                    },
                    {
                        "highlight_end": 11,
                        "highlight_start": 1,
                        "text": "         {"
                    },
                    {
                        "highlight_end": 46,
                        "highlight_start": 1,
                        "text": "             if ! (* left_val == * right_val)"
                    },
                    {
                        "highlight_end": 15,
                        "highlight_start": 1,
                        "text": "             {"
                    },
                    {
                        "highlight_end": 25,
                        "highlight_start": 1,
                        "text": "                 panic !"
                    },
                    {
                        "highlight_end": 57,
                        "highlight_start": 1,
                        "text": "                 (r#\"assertion failed: `(left == right)`"
                    },
                    {
                        "highlight_end": 16,
                        "highlight_start": 1,
                        "text": "  left: `{:?}`,"
                    },
                    {
                        "highlight_end": 22,
                        "highlight_start": 1,
                        "text": " right: `{:?}`: {}\"#,"
                    },
                    {
                        "highlight_end": 72,
                        "highlight_start": 1,
                        "text": "                  & * left_val, & * right_val, $ crate :: format_args !"
                    },
                    {
                        "highlight_end": 33,
                        "highlight_start": 1,
                        "text": "                  ($ ($ arg) +))"
                    },
                    {
                        "highlight_end": 15,
                        "highlight_start": 1,
                        "text": "             }"
                    },
                    {
                        "highlight_end": 11,
                        "highlight_start": 1,
                        "text": "         }"
                    },
                    {
                        "highlight_end": 7,
                        "highlight_start": 1,
                        "text": "     }"
                    },
                    {
                        "highlight_end": 6,
                        "highlight_start": 1,
                        "text": " }) ;"
                    }
                ]
            },
            "macro_decl_name": "assert_eq!",
            "span": {
                "byte_end": 38,
                "byte_start": 16,
                "column_end": 27,
                "column_start": 5,
                "expansion": null,
                "file_name": "src/main.rs",
                "is_primary": false,
                "label": null,
                "line_end": 2,
                "line_start": 2,
                "suggested_replacement": null,
                "suggestion_applicability": null,
                "text": [
                    {
                        "highlight_end": 27,
                        "highlight_start": 5,
                        "text": "    assert_eq!(1, \"love\");"
                    }
                ]
            }
        },
        "file_name": "<::core::macros::assert_eq macros>",
        "is_primary": true,
        "label": "no implementation for `{integer} == &str`",
        "line_end": 7,
        "line_start": 7,
        "suggested_replacement": null,
        "suggestion_applicability": null,
        "text": [
            {
                "highlight_end": 33,
                "highlight_start": 31,
                "text": "             if ! (* left_val == * right_val)"
            }
        ]
    }
]
}"##,
    );

    let workspace_root = PathBuf::from("/test/");
    let diag = map_rust_diagnostic_to_lsp(&diag, &workspace_root).expect("couldn't map diagnostic");
    insta::assert_debug_snapshot!(diag);
}

#[test]
#[cfg(not(windows))]
fn snap_macro_compiler_error() {
    let diag = parse_diagnostic(
        r##"{
    "rendered": "error: Please register your known path in the path module\n   --> crates/ra_hir_def/src/path.rs:265:9\n    |\n265 |         compile_error!(\"Please register your known path in the path module\")\n    |         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n    | \n   ::: crates/ra_hir_def/src/data.rs:80:16\n    |\n80  |     let path = path![std::future::Future];\n    |                -------------------------- in this macro invocation\n\n",
    "children": [],
    "code": null,
    "level": "error",
    "message": "Please register your known path in the path module",
    "spans": [
        {
            "byte_end": 8285,
            "byte_start": 8217,
            "column_end": 77,
            "column_start": 9,
            "expansion": {
                "def_site_span": {
                    "byte_end": 8294,
                    "byte_start": 7858,
                    "column_end": 2,
                    "column_start": 1,
                    "expansion": null,
                    "file_name": "crates/ra_hir_def/src/path.rs",
                    "is_primary": false,
                    "label": null,
                    "line_end": 267,
                    "line_start": 254,
                    "suggested_replacement": null,
                    "suggestion_applicability": null,
                    "text": [
                        {
                            "highlight_end": 28,
                            "highlight_start": 1,
                            "text": "macro_rules! __known_path {"
                        },
                        {
                            "highlight_end": 37,
                            "highlight_start": 1,
                            "text": "    (std::iter::IntoIterator) => {};"
                        },
                        {
                            "highlight_end": 33,
                            "highlight_start": 1,
                            "text": "    (std::result::Result) => {};"
                        },
                        {
                            "highlight_end": 29,
                            "highlight_start": 1,
                            "text": "    (std::ops::Range) => {};"
                        },
                        {
                            "highlight_end": 33,
                            "highlight_start": 1,
                            "text": "    (std::ops::RangeFrom) => {};"
                        },
                        {
                            "highlight_end": 33,
                            "highlight_start": 1,
                            "text": "    (std::ops::RangeFull) => {};"
                        },
                        {
                            "highlight_end": 31,
                            "highlight_start": 1,
                            "text": "    (std::ops::RangeTo) => {};"
                        },
                        {
                            "highlight_end": 40,
                            "highlight_start": 1,
                            "text": "    (std::ops::RangeToInclusive) => {};"
                        },
                        {
                            "highlight_end": 38,
                            "highlight_start": 1,
                            "text": "    (std::ops::RangeInclusive) => {};"
                        },
                        {
                            "highlight_end": 27,
                            "highlight_start": 1,
                            "text": "    (std::ops::Try) => {};"
                        },
                        {
                            "highlight_end": 22,
                            "highlight_start": 1,
                            "text": "    ($path:path) => {"
                        },
                        {
                            "highlight_end": 77,
                            "highlight_start": 1,
                            "text": "        compile_error!(\"Please register your known path in the path module\")"
                        },
                        {
                            "highlight_end": 7,
                            "highlight_start": 1,
                            "text": "    };"
                        },
                        {
                            "highlight_end": 2,
                            "highlight_start": 1,
                            "text": "}"
                        }
                    ]
                },
                "macro_decl_name": "$crate::__known_path!",
                "span": {
                    "byte_end": 8427,
                    "byte_start": 8385,
                    "column_end": 51,
                    "column_start": 9,
                    "expansion": {
                        "def_site_span": {
                            "byte_end": 8611,
                            "byte_start": 8312,
                            "column_end": 2,
                            "column_start": 1,
                            "expansion": null,
                            "file_name": "crates/ra_hir_def/src/path.rs",
                            "is_primary": false,
                            "label": null,
                            "line_end": 277,
                            "line_start": 270,
                            "suggested_replacement": null,
                            "suggestion_applicability": null,
                            "text": [
                                {
                                    "highlight_end": 22,
                                    "highlight_start": 1,
                                    "text": "macro_rules! __path {"
                                },
                                {
                                    "highlight_end": 43,
                                    "highlight_start": 1,
                                    "text": "    ($start:ident $(:: $seg:ident)*) => ({"
                                },
                                {
                                    "highlight_end": 51,
                                    "highlight_start": 1,
                                    "text": "        $crate::__known_path!($start $(:: $seg)*);"
                                },
                                {
                                    "highlight_end": 87,
                                    "highlight_start": 1,
                                    "text": "        $crate::path::ModPath::from_simple_segments($crate::path::PathKind::Abs, vec!["
                                },
                                {
                                    "highlight_end": 76,
                                    "highlight_start": 1,
                                    "text": "            $crate::path::__name![$start], $($crate::path::__name![$seg],)*"
                                },
                                {
                                    "highlight_end": 11,
                                    "highlight_start": 1,
                                    "text": "        ])"
                                },
                                {
                                    "highlight_end": 8,
                                    "highlight_start": 1,
                                    "text": "    });"
                                },
                                {
                                    "highlight_end": 2,
                                    "highlight_start": 1,
                                    "text": "}"
                                }
                            ]
                        },
                        "macro_decl_name": "path!",
                        "span": {
                            "byte_end": 2966,
                            "byte_start": 2940,
                            "column_end": 42,
                            "column_start": 16,
                            "expansion": null,
                            "file_name": "crates/ra_hir_def/src/data.rs",
                            "is_primary": false,
                            "label": null,
                            "line_end": 80,
                            "line_start": 80,
                            "suggested_replacement": null,
                            "suggestion_applicability": null,
                            "text": [
                                {
                                    "highlight_end": 42,
                                    "highlight_start": 16,
                                    "text": "    let path = path![std::future::Future];"
                                }
                            ]
                        }
                    },
                    "file_name": "crates/ra_hir_def/src/path.rs",
                    "is_primary": false,
                    "label": null,
                    "line_end": 272,
                    "line_start": 272,
                    "suggested_replacement": null,
                    "suggestion_applicability": null,
                    "text": [
                        {
                            "highlight_end": 51,
                            "highlight_start": 9,
                            "text": "        $crate::__known_path!($start $(:: $seg)*);"
                        }
                    ]
                }
            },
            "file_name": "crates/ra_hir_def/src/path.rs",
            "is_primary": true,
            "label": null,
            "line_end": 265,
            "line_start": 265,
            "suggested_replacement": null,
            "suggestion_applicability": null,
            "text": [
                {
                    "highlight_end": 77,
                    "highlight_start": 9,
                    "text": "        compile_error!(\"Please register your known path in the path module\")"
                }
            ]
        }
    ]
}
        "##,
    );

    let workspace_root = PathBuf::from("/test/");
    let diag = map_rust_diagnostic_to_lsp(&diag, &workspace_root).expect("couldn't map diagnostic");
    insta::assert_debug_snapshot!(diag);
}
