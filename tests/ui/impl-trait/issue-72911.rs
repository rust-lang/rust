// Regression test for #72911.

pub struct Lint {}

impl Lint {}

pub fn gather_all() -> impl Iterator<Item = Lint> {
    lint_files().flat_map(|f| gather_from_file(&f))
}

fn gather_from_file(dir_entry: &foo::MissingItem) -> impl Iterator<Item = Lint> {
    //~^ ERROR: failed to resolve
    unimplemented!()
}

fn lint_files() -> impl Iterator<Item = foo::MissingItem> {
    //~^ ERROR: failed to resolve
    unimplemented!()
}

fn main() {}
