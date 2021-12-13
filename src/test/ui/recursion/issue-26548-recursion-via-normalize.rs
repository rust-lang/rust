trait Mirror { //~ NOTE required by a bound in this
    type It: ?Sized;
}
impl<T: ?Sized> Mirror for T {
    type It = Self;
}
struct S(Option<<S as Mirror>::It>);
//~^ ERROR overflow evaluating the requirement `S: Sized`
//~| NOTE required because it appears within the type `S`
//~| NOTE required by a bound in `Option`

fn main() {
    let _s = S(None);
}
