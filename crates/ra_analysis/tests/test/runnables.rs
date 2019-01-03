use test_utils::assert_eq_dbg;

use ra_analysis::mock_analysis::analysis_and_position;

#[test]
fn test_runnables() {
    let (analysis, pos) = analysis_and_position(
        r#"
        //- /lib.rs
        <|> //empty
        fn main() {}

        #[test]
        fn test_foo() {}

        #[test]
        #[ignore]
        fn test_foo() {}
        "#,
    );
    let runnables = analysis.runnables(pos.file_id).unwrap();
    assert_eq_dbg(
        r#"[Runnable { range: [1; 21), kind: Bin },
                Runnable { range: [22; 46), kind: Test { name: "test_foo" } },
                Runnable { range: [47; 81), kind: Test { name: "test_foo" } }]"#,
        &runnables,
    )
}

#[test]
fn test_runnables_module() {
    let (analysis, pos) = analysis_and_position(
        r#"
        //- /lib.rs
        <|> //empty
        mod test_mod {
            #[test]
            fn test_foo1() {}
        }
        "#,
    );
    let runnables = analysis.runnables(pos.file_id).unwrap();
    assert_eq_dbg(
        r#"[Runnable { range: [1; 59), kind: TestMod { path: "test_mod" } },
                Runnable { range: [28; 57), kind: Test { name: "test_foo1" } }]"#,
        &runnables,
    )
}

#[test]
fn test_runnables_one_depth_layer_module() {
    let (analysis, pos) = analysis_and_position(
        r#"
        //- /lib.rs
        <|> //empty
        mod foo {
            mod test_mod {
                #[test]
                fn test_foo1() {}
            }
        }
        "#,
    );
    let runnables = analysis.runnables(pos.file_id).unwrap();
    assert_eq_dbg(
        r#"[Runnable { range: [23; 85), kind: TestMod { path: "foo::test_mod" } },
                Runnable { range: [46; 79), kind: Test { name: "test_foo1" } }]"#,
        &runnables,
    )
}

#[test]
fn test_runnables_multiple_depth_module() {
    let (analysis, pos) = analysis_and_position(
        r#"
        //- /lib.rs
        <|> //empty
        mod foo {
            mod bar {
                mod test_mod {
                    #[test]
                    fn test_foo1() {}
                }
            }
        }
        "#,
    );
    let runnables = analysis.runnables(pos.file_id).unwrap();
    assert_eq_dbg(
        r#"[Runnable { range: [41; 115), kind: TestMod { path: "foo::bar::test_mod" } },
                Runnable { range: [68; 105), kind: Test { name: "test_foo1" } }]"#,
        &runnables,
    )
}

#[test]
fn test_runnables_no_test_function_in_module() {
    let (analysis, pos) = analysis_and_position(
        r#"
        //- /lib.rs
        <|> //empty
        mod test_mod {
            fn foo1() {}
        }
        "#,
    );
    let runnables = analysis.runnables(pos.file_id).unwrap();
    assert_eq_dbg(r#"[]"#, &runnables)
}
