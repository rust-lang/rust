// compile-flags: -Z parse-only

static s: &'static str =
    r#"
      "## //~ ERROR expected one of `.`, `;`, `?`, or an operator, found `#`
;
