pub trait Insertable {
    type Values;

    fn values(&self) -> Self::Values;
}

impl<T> Insertable for Option<T> {
    type Values = ();

    fn values(self) -> Self::Values {
        //~^ ERROR method `values` has an incompatible type for trait
        self.map(Insertable::values).unwrap_or_default()
        //~^ ERROR type mismatch in function arguments
    }
}

fn main() {}
