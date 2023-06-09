use super::check;

#[test]
fn function_return_type_mismatch_1() {
    check(
        r#"
fn test() -> &'static str {
    5
  //^ expected &str, got i32
}
"#,
    );
}

#[test]
fn function_return_type_mismatch_2() {
    check(
        r#"
fn test(x: bool) -> &'static str {
    if x {
        return 1;
             //^ expected &str, got i32
    }
    "ok"
}
"#,
    );
}

#[test]
fn function_return_type_mismatch_3() {
    check(
        r#"
fn test(x: bool) -> &'static str {
    if x {
        return "ok";
    }
    1
  //^ expected &str, got i32
}
"#,
    );
}

#[test]
fn function_return_type_mismatch_4() {
    check(
        r#"
fn test(x: bool) -> &'static str {
    if x {
        "ok"
    } else {
        1
      //^ expected &str, got i32
    }
}
"#,
    );
}

#[test]
fn function_return_type_mismatch_5() {
    check(
        r#"
fn test(x: bool) -> &'static str {
    if x {
        1
      //^ expected &str, got i32
    } else {
        "ok"
    }
}
"#,
    );
}

#[test]
fn non_unit_block_expr_stmt_no_semi() {
    check(
        r#"
fn test(x: bool) {
    if x {
        "notok"
      //^^^^^^^ expected (), got &str
    } else {
        "ok"
      //^^^^ expected (), got &str
    }
    match x { true => true, false => 0 }
  //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ expected (), got bool
                                   //^ expected bool, got i32
    ()
}
"#,
    );
}
