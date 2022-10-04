//~ ERROR  cycle detected when computing layout of `core::option::Option<<S as Mirror>::It>`
//~| NOTE ...which requires computing layout of `core::option::Option<S>`...
//~| NOTE ...which requires computing layout of `S`...
//~| NOTE ...which again requires computing layout of `core::option::Option<<S as Mirror>::It>`, completing the cycle

trait Mirror {
    //~^ NOTE cycle used when checking deathness of variables in top-level module
    type It: ?Sized;
}
impl<T: ?Sized> Mirror for T {
    type It = Self;
}
struct S(Option<<S as Mirror>::It>);

fn main() {
    let _s = S(None);
}
