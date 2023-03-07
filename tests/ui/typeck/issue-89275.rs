#![recursion_limit = "5"] // To reduce noise

//expect mutability error when ambiguous traits are in scope
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

fn downcast<'a, W: ?Sized>() -> &'a W {
    todo!()
}

struct Other;

fn main() {
    let other: &mut Other = downcast();//~ERROR 28:29: 28:39: mismatched types [E0308]
}
