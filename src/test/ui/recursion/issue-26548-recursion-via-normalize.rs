//~ ERROR cycle detected when computing layout of `S`
//~| NOTE ...which requires computing layout of `core::option::Option<<S as Mirror>::It>`...
//~| NOTE ...which requires computing layout of `core::option::Option<S>`...
//~| NOTE ...which again requires computing layout of `S`, completing the cycle
//~| NOTE cycle used when computing layout of `core::option::Option<S>`

// build-fail

trait Mirror {
    type It: ?Sized;
}
impl<T: ?Sized> Mirror for T {
    type It = Self;
}
struct S(Option<<S as Mirror>::It>);

fn main() {
    let _s = S(None);
}
