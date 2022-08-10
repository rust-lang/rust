#![recursion_limit = "5"] // To reduce noise

//expect incompatible type error when ambiguous traits are in scope
//and not an overflow error on the span in the main function.

struct Ratio<T>(T);

pub trait Pow {
    fn pow(self) -> Self;
}

impl<'a, T> Pow for &'a Ratio<T>
where
    &'a T: Pow,
{
    fn pow(self) -> Self {
        self
    }
}

fn downcast<'a, W: ?Sized>() -> std::io::Result<&'a W> {
    todo!()
}

struct Other;

fn main() -> std::io::Result<()> {
    let other: Other = downcast()?;//~ERROR 28:24: 28:35: `?` operator has incompatible types
    Ok(())
}
