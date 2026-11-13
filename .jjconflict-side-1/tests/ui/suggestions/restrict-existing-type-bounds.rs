pub trait TryAdd<Rhs = Self> {
    type Error;
    type Output;

    fn try_add(self, rhs: Rhs) -> Result<Self::Output, Self::Error>;
}

impl<T: TryAdd> TryAdd for Option<T> {
    type Error = <T as TryAdd>::Error;
    type Output = Option<<T as TryAdd>::Output>;

    fn try_add(self, rhs: Self) -> Result<Self::Output, Self::Error> {
        Ok(self) //~ ERROR mismatched types
    }
}

struct Other<A>(A);

struct X;

impl<T: TryAdd<Error = X>> TryAdd for Other<T> {
    type Error = <T as TryAdd>::Error;
    type Output = Other<<T as TryAdd>::Output>;

    fn try_add(self, rhs: Self) -> Result<Self::Output, Self::Error> {
        Ok(self) //~ ERROR mismatched types
    }
}

fn main() {}
