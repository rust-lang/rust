static s: &'static str =
    r#""## //~ ERROR too many `#` when terminating raw string
;

static s2: &'static str =
    r#"
      "#### //~ ERROR too many `#` when terminating raw string
;

const A: &'static str = r"" //~ ERROR expected `;`, found `#`

// Test
#[test]
fn test() {}

const B: &'static str = r""## //~ ERROR too many `#` when terminating raw string

// Test
#[test]
fn test2() {}

fn main() {}
