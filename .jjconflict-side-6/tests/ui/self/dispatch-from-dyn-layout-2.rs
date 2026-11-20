//@ check-pass
// Regression test for #90110.

// Make sure that object safety checking doesn't freak out when
// we have impossible-to-satisfy `Sized` predicates.

trait Parser
where
    for<'a> (dyn Parser + 'a): Sized,
{
    fn parse_line(&self);
}

fn foo(_: &dyn Parser) {}

fn main() {}
