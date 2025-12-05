//@ check-pass

trait Trait<T> {
    const ASSOC_CONST: usize = 0;
}

impl Trait<()> for u8 {}

// `u8::ASSOC_CONST` is resolved today, but will be ambiguous
// under lazy normalization.
fn foo<T, U>() -> [(T, U); u8::ASSOC_CONST]
where
    u8: Trait<T> + Trait<U>,
{
    todo!()
}

fn main() {}
