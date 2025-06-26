#[derive(Default)]
pub struct Derive;

pub struct Manual;

impl Default for Manual {
    fn default() -> Self {
        Self
    }
}

//@ arg default [.index[] | select(.inner.impl.trait?.path == "Default")]
//@ eq $default[] | select(.inner.impl.for?.resolved_path?.path == "Derive").attrs | ., ["#[automatically_derived]"]
//@ eq $default[] | select(.inner.impl.for?.resolved_path?.path == "Manual").attrs | ., []
