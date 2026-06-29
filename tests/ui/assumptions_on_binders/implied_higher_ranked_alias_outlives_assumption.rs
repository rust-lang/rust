//@ compile-flags: -Znext-solver -Zassumptions-on-binders
//@ check-pass

#![feature(generic_const_items)]

// sorry for writing this
// - boxy

// for<'a> where(for<'d> <T as Trait<'a, 'd>::Assoc: 'c) {
//     for<'b> {
//         <T as Trait<'a_u1, 'b_u2>>::Assoc: 'c
//     }
//
//     rewritten to: for<'b> <T as Trait<'a_u1, '^b>>::Assoc: 'c
// }
// rewritten to: true (via assumption)
// rewritting to `for<'a, 'b> <T as Trait<^0, ^1>>::Assoc: 'c` would be wrong

trait Trait<'a, 'b> {
    type Assoc;
}

struct ImpliedBound<'a, 'c, T: for<'b> Trait<'a, 'b>>(T, &'a (), &'c ())
where
    for<'b> <T as Trait<'a, 'b>>::Assoc: 'c,;

trait InnerBinder<'a, 'b, 'c> {}
impl<'a, 'b, 'c, S> InnerBinder<'a, 'b, 'c> for S
where
    S: Trait<'a, 'b>,
    <S as Trait<'a, 'b>>::Assoc: 'c {}

trait OuterBinder<'a, 'c, T0> {}
impl<'a, 'c, T0, S> OuterBinder<'a, 'c, T0> for S
where
    for<'b> S: InnerBinder<'a, 'b, 'c>, {}

struct ReqTrait<'c, T>(&'c (), T)
where
    for<'a> T: OuterBinder<'a, 'c, ImpliedBound<'a, 'c, T>>,;

fn borrowck_env<'c, T>()
where
    T: for<'a, 'b> Trait<'a, 'b>
{
    let _: ReqTrait<'c, T>;
}

const REGIONCK_ENV<'c, T>: ReqTrait<'c, T> = todo!()
where
    T: for<'a, 'b> Trait<'a, 'b>;

fn main() {}
