// build-pass (FIXME(62277): could be check-pass?)

// this tests the `unknown_lint` lint, especially the suggestions

// the suggestion only appears if a lint with the lowercase name exists
#[allow(FOO_BAR)]
// the suggestion appears on all-uppercase names
#[warn(DEAD_CODE)]
// the suggestion appears also on mixed-case names
#[deny(Warnings)]
fn main() {
    unimplemented!();
}
