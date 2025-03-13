// Check that we emit the `CONST_EVALUATABLE_UNCHECKED` FCW when a repeat expr
// count depends on in-scope where clauses to prove an impossible bound but does
// not otherwise depend on any in-scope generic parameters.

pub trait Unimplemented<'a> {}

trait Trait<T>
where
    for<'a> T: Unimplemented<'a>,
{
    const ASSOC: usize;
}

impl<T> Trait<T> for u8
where
    for<'a> T: Unimplemented<'a>,
{
    const ASSOC: usize = 1;
}

pub fn impossible_preds_repeat_expr_count()
where
    for<'a> (): Unimplemented<'a>,
{
    let _a: [(); 1] = [(); <u8 as Trait<()>>::ASSOC];
    //~^ WARN: cannot use constants which depend on trivially-false where clauses
    //~| WARN: this was previously accepted by the compiler
    //~^^^ ERROR: mismatched types
}

struct Foo<const N: usize>;

pub fn impossible_preds_normal_arg()
where
    for<'a> (): Unimplemented<'a>,
{
    let _a: Foo<1> = Foo::<{ <u8 as Trait<()>>::ASSOC }>;
    //~^ ERROR: the trait bound `for<'a> (): Unimplemented<'a>`
    //~| ERROR: the trait bound `for<'a> (): Unimplemented<'a>`
}

fn main() {}
