#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

fn foo<const N: usize>(
    _: [u8; {
        {
            N
        }
    }],
) {
}

fn ice<const L: usize>()
where
    [(); (L - 1) + 1 + L]:,
{
    foo::<_, L>([(); L + 1 + L]);
    //~^ ERROR: mismatched types
    //~^^ ERROR: unconstrained generic constant
    //~^^^ ERROR: function takes 1 generic argument but 2 generic arguments were supplied
    //~^^^^ ERROR: unconstrained generic constant
    //~^^^^^ ERROR: unconstrained generic constant `L + 1 + L`
    //~^^^^^^ ERROR: unconstrained generic constant `L + 1`
}

fn main() {}
