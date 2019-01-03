use ra_analysis::mock_analysis::single_file_with_range;

#[test]
fn test_type_of_for_function() {
    let (analysis, range) = single_file_with_range(
        "
        pub fn foo() -> u32 { 1 };

        fn main() {
            let foo_test = <|>foo()<|>;
        }
        ",
    );

    let type_name = analysis.type_of(range).unwrap().unwrap();
    assert_eq!("u32", &type_name);
}

// FIXME: improve type_of to make this work
#[test]
fn test_type_of_for_num() {
    let (analysis, range) = single_file_with_range(
        r#"
        fn main() {
            let foo_test = <|>"foo"<|>;
        }
        "#,
    );

    assert!(analysis.type_of(range).unwrap().is_none());
}
// FIXME: improve type_of to make this work
#[test]
fn test_type_of_for_binding() {
    let (analysis, range) = single_file_with_range(
        "
        pub fn foo() -> u32 { 1 };

        fn main() {
            let <|>foo_test<|> = foo();
        }
        ",
    );

    assert!(analysis.type_of(range).unwrap().is_none());
}

// FIXME: improve type_of to make this work
#[test]
fn test_type_of_for_expr_1() {
    let (analysis, range) = single_file_with_range(
        "
        fn main() {
            let foo = <|>1 + foo_test<|>;
        }
        ",
    );

    let type_name = analysis.type_of(range).unwrap().unwrap();
    assert_eq!("[unknown]", &type_name);
}

// FIXME: improve type_of to make this work
#[test]
fn test_type_of_for_expr_2() {
    let (analysis, range) = single_file_with_range(
        "
        fn main() {
            let foo: usize = 1;
            let bar = <|>1 + foo_test<|>;
        }
        ",
    );

    let type_name = analysis.type_of(range).unwrap().unwrap();
    assert_eq!("[unknown]", &type_name);
}
