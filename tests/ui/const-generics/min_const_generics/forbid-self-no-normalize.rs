trait AlwaysApplicable {
    type Assoc;
}
impl<T: ?Sized> AlwaysApplicable for T {
    type Assoc = usize;
}

trait BindsParam<T> {
    type ArrayTy;
}
impl<T> BindsParam<T> for <T as AlwaysApplicable>::Assoc {
    type ArrayTy = [u8; Self::MAX]; //~ ERROR generic `Self` types
}

fn main() {}
