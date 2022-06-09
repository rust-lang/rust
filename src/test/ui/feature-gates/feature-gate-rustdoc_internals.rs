#[doc(keyword = "match")] //~ ERROR: `#[doc(keyword)]` is meant for internal use only
/// wonderful
mod foo {}

trait Mine {}

#[doc(tuple_varadic)]  //~ ERROR: `#[doc(tuple_varadic)]` is meant for internal use only
impl<T> Mine for (T,) {}

fn main() {}
