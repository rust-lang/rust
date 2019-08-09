// compile-fail

#![feature(associated_type_bounds)]

trait Tr1 { type As1; }
trait Tr2 { type As2; }

struct St<'a, 'b, T: Tr1<As1: Tr2>> { // `T: 'b` is *not* implied!
    f0: &'a T, // `T: 'a` is implied.
    f1: &'b <T::As1 as Tr2>::As2, // `<T::As1 as Tr2>::As2: 'a` is implied.
}

fn _bad_st<'a, 'b, T>(x: St<'a, 'b, T>)
where
    T: Tr1,
    T::As1: Tr2,
{
    // This should fail because `T: 'b` is not implied from `WF(St<'a, 'b, T>)`.
    let _failure_proves_not_implied_outlives_region_b: &'b T = &x.f0;
    //~^ ERROR lifetime mismatch [E0623]
}

enum En7<'a, 'b, T> // `<T::As1 as Tr2>::As2: 'a` is implied.
where
    T: Tr1,
    T::As1: Tr2,
{
    V0(&'a T),
    V1(&'b <T::As1 as Tr2>::As2),
}

fn _bad_en7<'a, 'b, T>(x: En7<'a, 'b, T>)
where
    T: Tr1,
    T::As1: Tr2,
{
    match x {
        En7::V0(x) => {
            // Also fails for the same reason as above:
            let _failure_proves_not_implied_outlives_region_b: &'b T = &x;
            //~^ ERROR lifetime mismatch [E0623]
        },
        En7::V1(_) => {},
    }
}

fn main() {}
