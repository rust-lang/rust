trait Mirror {
    type It: ?Sized;
}
impl<T: ?Sized> Mirror for T {
    type It = Self;
}
struct S(Option<<S as Mirror>::It>);
//~^ ERROR overflow evaluating the requirement `S: Sized`
//~| NOTE required because it appears within the type `S`
//~| NOTE type parameters have an implicit `Sized` obligation

fn main() {
    let _s = S(None);
}
