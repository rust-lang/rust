trait DynEq {}

impl<'a> PartialEq for &'a (dyn DynEq + 'static) {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl Eq for &dyn DynEq {} //~ ERROR E0308

fn main() {
}
