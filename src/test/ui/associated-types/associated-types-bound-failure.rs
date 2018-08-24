// Test equality constraints on associated types in a where clause.

pub trait ToInt {
    fn to_int(&self) -> isize;
}

pub trait GetToInt
{
    type R;

    fn get(&self) -> <Self as GetToInt>::R;
}

fn foo<G>(g: G) -> isize
    where G : GetToInt
{
    ToInt::to_int(&g.get()) //~ ERROR E0277
}

fn bar<G : GetToInt>(g: G) -> isize
    where G::R : ToInt
{
    ToInt::to_int(&g.get()) // OK
}

pub fn main() {
}
