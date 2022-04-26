// check-pass

#[derive(Clone)]
struct Constraint;

fn constraints<C>(constraints: C)
where C: Into<Vec<Constraint>>
{
    let _: Vec<Constraint> = constraints.into();
}

fn main() {
    constraints(vec![Constraint].as_ref());
}
