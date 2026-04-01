#![feature(rustc_attrs)]

#[rustc_must_implement_one_of(eq, neq)]
trait Equal {
    fn eq(&self, other: &Self) -> bool {
        !self.neq(other)
    }

    fn neq(&self, other: &Self) -> bool {
        !self.eq(other)
    }
}

struct T0;
struct T1;
struct T2;
struct T3;

impl Equal for T0 {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl Equal for T1 {
    fn neq(&self, _other: &Self) -> bool {
        false
    }
}

impl Equal for T2 {
    fn eq(&self, _other: &Self) -> bool {
        true
    }

    fn neq(&self, _other: &Self) -> bool {
        false
    }
}

impl Equal for T3 {}
//~^ ERROR not all trait items implemented, missing one of: `eq`, `neq`

fn main() {}
