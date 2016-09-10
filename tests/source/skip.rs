// Test the skip attribute works

#[rustfmt_skip]
fn foo() { badly; formatted; stuff
; }

#[rustfmt_skip]
trait Foo
{
fn foo(
);
}

impl LateLintPass for UsedUnderscoreBinding {
    #[cfg_attr(rustfmt, rustfmt_skip)]
    fn check_expr() { // comment
    }
}
