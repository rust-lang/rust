#[doc(keyword = "match")] //~ ERROR: `#[doc(keyword)]` is meant for internal use only
/// wonderful
mod foo {}

trait Mine {}

#[doc(tuple_variadic)]  //~ ERROR: `#[doc(tuple_variadic)]` is meant for internal use only
impl<T> Mine for (T,) {}

fn main() {}
