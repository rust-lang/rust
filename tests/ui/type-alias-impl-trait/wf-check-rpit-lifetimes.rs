//@check-pass

pub struct Key;
#[derive(Clone)]
pub struct Value;

use std::collections::HashMap;

pub struct DiagnosticBuilder<'db> {
    inner: HashMap<&'db Key, Vec<&'db Value>>,
}

impl<'db> DiagnosticBuilder<'db> {
    pub fn iter<'a>(&'a self) -> impl Iterator<Item = (&'db Key, impl Iterator<Item = &'a Value>)> {
        self.inner.iter().map(|(key, values)| (*key, values.iter().map(|v| *v)))
    }
}

fn main() {}
