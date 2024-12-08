// Regression test for #131915 where we did not handle macro calls as
// statements correctly when determining if a const argument should
// have a `DefId` created or not.

macro_rules! y {
    ( $($matcher:tt)*) => {
        x
        //~^ ERROR: cannot find value `x` in this scope
    };
}

const _: A<
    //~^ ERROR: free constant item without body
    //~| ERROR: cannot find type `A` in this scope
    {
        y! { test.tou8 }
    },
>;

fn main() {}
