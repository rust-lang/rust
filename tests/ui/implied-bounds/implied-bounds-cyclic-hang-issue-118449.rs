// Regression test for #118449.

trait HasParent {
    type Parent: HasParent;
}

struct Visitor<'a, Type>
//~^ ERROR overflow computing implied lifetime bounds
where
    Type: HasParent,
{
    parent: &'a Visitor<'a, Type::Parent>,
    current: &'a Type,
}

fn main() {}
