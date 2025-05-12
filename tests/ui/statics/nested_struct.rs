//@ check-pass
/// oli-obk added this test after messing up the interner logic
/// around mutability of nested allocations. This was not caught
/// by the test suite, but by trying to build stage2 rustc.
/// There is no real explanation for this test, as it was just
/// a bug during a refactoring.

pub struct Lint {
    pub name: &'static str,
    pub desc: &'static str,
    pub report_in_external_macro: bool,
    pub is_externally_loaded: bool,
    pub crate_level_only: bool,
}

static FOO: &Lint = &Lint {
    name: &"foo",
    desc: "desc",
    report_in_external_macro: false,
    is_externally_loaded: true,
    crate_level_only: false,
};

fn main() {}
