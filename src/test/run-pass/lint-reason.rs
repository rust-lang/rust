#[deny(no_lint_reason, reason="")];

#[allow(no_lint_reason, reason="")]
mod foo {
    #[warn(unused_unsafe)];
}

fn main() { }
