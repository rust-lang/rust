// rustfmt-match_arm_leading_pipes: Preserve

fn foo() {
    match foo {
        | "foo"      | "bar" => {}
        | "baz"
        |                   "something relatively long"
        | "something really really really realllllllllllllly long" => println!("x"),
        | "qux" =>           println!("y"),
        _ => {}
    }
}

fn issue_3973() {
    match foo {
        | "foo"
        | "bar" => {}
        _ => {}
    }
}

fn bar() {
    match baz {
        "qux"     => {    }
        "foo" | "bar" => {}
        _ => {}
    }
}

fn f(x: NonAscii) -> bool {
    match x {
      // foo
              |   Éfgh => true,
        _ => false,
    }
}