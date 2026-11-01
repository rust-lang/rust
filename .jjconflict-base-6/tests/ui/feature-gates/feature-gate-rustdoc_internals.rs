#[doc(keyword = "match")] //~ ERROR: this subset of the `doc` attribute is meant for internal use only
/// wonderful
const _: () = ();

#[doc(attribute = "repr")] //~ ERROR this subset of the `doc` attribute is meant for internal use only
/// wonderful
const _: () = ();

trait Mine {}

#[doc(fake_variadic)]  //~ ERROR this subset of the `doc` attribute is meant for internal use only
impl<T> Mine for (T,) {}

#[doc(search_unbox)]  //~ ERROR this subset of the `doc` attribute is meant for internal use only
struct Wrap<T> (T);

fn main() {
    #[doc(search_unbox)]
    //~^ ERROR this subset of the `doc` attribute is meant for internal use only [E0658]
    println!();

    #[doc(fake_variadic)]
    //~^ ERROR this subset of the `doc` attribute is meant for internal use only [E0658]
    println!();

    #[doc(attribute = "repr")]
    //~^ ERROR this subset of the `doc` attribute is meant for internal use only [E0658]
    println!();
}
