#[repr(packed)]
pub struct Bad<T: ?Sized> {
    data: T, //~ ERROR the size for values of type
}

fn main() {}
