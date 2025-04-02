#[derive(Default)]
pub struct Derive;

pub struct Manual;

impl Default for Manual {
    fn default() -> Self {
        Self
    }
}

//@ is '$.index[?(@.inner.impl.for.resolved_path.path == "Derive" && @.inner.impl.trait.path == "Default")].attrs' '["#[automatically_derived]"]'
//@ is '$.index[?(@.inner.impl.for.resolved_path.path == "Manual" && @.inner.impl.trait.path == "Default")].attrs' '[]'
