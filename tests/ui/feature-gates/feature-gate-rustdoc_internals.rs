#[doc(keyword = "match")] //~ ERROR: `#[doc(keyword)]` is meant for internal use only
/// wonderful
mod foo {}

#[doc(attribute = "repr")] //~ ERROR: `#[doc(attribute)]` is meant for internal use only
/// wonderful
mod foo2 {}

trait Mine {}

#[doc(fake_variadic)]  //~ ERROR: `#[doc(fake_variadic)]` is meant for internal use only
impl<T> Mine for (T,) {}

#[doc(search_unbox)]  //~ ERROR: `#[doc(search_unbox)]` is meant for internal use only
struct Wrap<T> (T);

fn main() {}
