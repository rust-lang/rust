#![feature(rustc_attrs)]
#![feature(staged_api)]
#![stable(feature="s", since="1.0.0")]

#[rustc_must_implement_one_of(a1, b1)]
#[stable(feature="s", since="1.0.0")]
pub trait Trait1 {
    #[stable(feature="s", since="1.0.0")]
    fn a1(&self) -> u64 {
        self.a1() + 1
    }

    #[stable(feature="s", since="1.0.0")]
    fn b1(&self) -> u64 {
        self.b1() + 1
    }
}

#[rustc_must_implement_one_of(a2, b2)]
#[stable(feature="s", since="1.0.0")]
pub trait Trait2 {
    #[stable(feature="s", since="1.0.0")]
    fn a2(&self) -> u64 {
        self.b2() + 1
    }

    #[unstable(feature="trait2_b2", issue="none")]
    fn b2(&self) -> u64 {
        self.a2() + 1
    }
}
