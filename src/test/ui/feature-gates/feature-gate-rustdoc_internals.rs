#[doc(keyword = "match")] //~ ERROR: `#[doc(keyword)]` is meant for internal use only
/// wonderful
mod foo {}

trait Mine {}

#[doc(fake_variadic)]  //~ ERROR: `#[doc(fake_variadic)]` is meant for internal use only
impl<T> Mine for (T,) {}

fn main() {}
