#[derive(Default)]
pub struct Derive;

pub struct Manual;

impl Default for Manual {
    fn default() -> Self {
        Self
    }
}

//@ is '$.types[0].resolved_path.path' '"Derive"'
//@ is '$.types[14].resolved_path.path' '"Manual"'

//@ is '$.index[?(@.inner.impl.for == 0 && @.inner.impl.trait.path == "Default")].attrs' '["#[automatically_derived]"]'
//@ is '$.index[?(@.inner.impl.for == 14 && @.inner.impl.trait.path == "Default")].attrs' '[]'
