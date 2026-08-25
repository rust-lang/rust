//@ revisions: current next
//@[next] compile-flags: -Znext-solver

#![feature(const_trait_impl)]

const trait Add<Rhs = Self> {
    type Output;

    fn add(self, other: Rhs) -> Self::Output;
}

const impl Add for i32 {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        self + other
    }
}

struct NonConstAdd(i32);

impl Add for NonConstAdd {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        NonConstAdd(self.0.add(rhs.0))
    }
}

const trait Foo {
    type Bar: [const] Add;
}

const impl Foo for NonConstAdd {
    type Bar = NonConstAdd;
    //~^ ERROR the trait bound `NonConstAdd: [const] Add` is not satisfied
}

const trait Baz {
    type Qux: Add;
}

const impl Baz for NonConstAdd {
    type Qux = NonConstAdd; // OK
}

fn main() {}
