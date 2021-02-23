pub macro_rules! m1 { () => {} } //~ ERROR `pub` on `macro_rules` items is unstable

#[cfg(FALSE)]
pub macro_rules! m2 { () => {} } //~ ERROR `pub` on `macro_rules` items is unstable

pub(crate) macro_rules! m3 { () => {} } //~ ERROR `pub` on `macro_rules` items is unstable

pub(in self) macro_rules! m4 { () => {} } //~ ERROR `pub` on `macro_rules` items is unstable

fn main() {}
