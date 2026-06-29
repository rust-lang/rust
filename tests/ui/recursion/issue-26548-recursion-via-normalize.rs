//~ NOTE cycle used when computing layout of `core::option::Option<<S as Mirror>::It>`
//~? ERROR cycle detected when computing layout of `core::option::Option<S>`
//~? NOTE for more information, see <https://rustc-dev-guide.rust-lang.org/overview.html#queries> and <https://rustc-dev-guide.rust-lang.org/query.html>
//~? NOTE ...which requires computing layout of `core::option::Option<<S as Mirror>::It>`...
//~? NOTE ...which again requires computing layout of `core::option::Option<S>`, completing the cycle

trait Mirror {
    type It: ?Sized;
}
impl<T: ?Sized> Mirror for T {
    type It = Self;
}
struct S(Option<<S as Mirror>::It>);
//~^ NOTE ...which requires computing layout of `S`...

fn main() {
    let _s = S(None);
}
