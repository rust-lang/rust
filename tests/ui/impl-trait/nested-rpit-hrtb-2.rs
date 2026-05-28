// The nested impl Trait references a higher-ranked region

trait Trait<'a> { type Assoc; }
impl<'a> Trait<'a> for () { type Assoc = &'a str; }

fn test() -> impl for<'a> Trait<'a, Assoc = impl Sized> {}
//~^ ERROR captures lifetime that does not appear in bounds

fn main() {}
