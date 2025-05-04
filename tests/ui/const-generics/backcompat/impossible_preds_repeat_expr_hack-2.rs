//@ check-pass

// Split out from `impossible_preds_repeat_expr_hack` to allow for delay span bugs to ICE

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
    // This won't actually be an error in the future but we can't really determine that
    let _b = [(); 1 + 1];
    //~^ WARN: cannot use constants which depend on trivially-false where clauses
    //~| WARN: this was previously accepted by the compiler
}

fn main() {}
