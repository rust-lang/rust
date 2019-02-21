// Check testing of equality constraints in a higher-ranked context.

pub trait TheTrait<T> {
    type A;

    fn get(&self, t: T) -> Self::A;
}

struct IntStruct {
    x: isize
}

impl<'a> TheTrait<&'a isize> for IntStruct {
    type A = &'a isize;

    fn get(&self, t: &'a isize) -> &'a isize {
        t
    }
}

struct UintStruct {
    x: isize
}

impl<'a> TheTrait<&'a isize> for UintStruct {
    type A = &'a usize;

    fn get(&self, t: &'a isize) -> &'a usize {
        panic!()
    }
}

struct Tuple {
}

impl<'a> TheTrait<(&'a isize, &'a isize)> for Tuple {
    type A = &'a isize;

    fn get(&self, t: (&'a isize, &'a isize)) -> &'a isize {
        t.0
    }
}

fn foo<T>()
    where T : for<'x> TheTrait<&'x isize, A = &'x isize>
{
    // ok for IntStruct, but not UintStruct
}

fn bar<T>()
    where T : for<'x> TheTrait<&'x isize, A = &'x usize>
{
    // ok for UintStruct, but not IntStruct
}

fn tuple_one<T>()
    where T : for<'x,'y> TheTrait<(&'x isize, &'y isize), A = &'x isize>
{
    // not ok for tuple, two lifetimes and we pick first
}

fn tuple_two<T>()
    where T : for<'x,'y> TheTrait<(&'x isize, &'y isize), A = &'y isize>
{
    // not ok for tuple, two lifetimes and we pick second
}

fn tuple_three<T>()
    where T : for<'x> TheTrait<(&'x isize, &'x isize), A = &'x isize>
{
    // ok for tuple
}

fn tuple_four<T>()
    where T : for<'x,'y> TheTrait<(&'x isize, &'y isize)>
{
    // not ok for tuple, two lifetimes, and lifetime matching is invariant
}

pub fn call_foo() {
    foo::<IntStruct>();
    foo::<UintStruct>(); //~ ERROR type mismatch
}

pub fn call_bar() {
    bar::<IntStruct>(); //~ ERROR type mismatch
    bar::<UintStruct>();
}

pub fn call_tuple_one() {
    tuple_one::<Tuple>();
    //~^ ERROR not satisfied
    //~| ERROR type mismatch
}

pub fn call_tuple_two() {
    tuple_two::<Tuple>();
    //~^ ERROR not satisfied
    //~| ERROR type mismatch
}

pub fn call_tuple_three() {
    tuple_three::<Tuple>();
}

pub fn call_tuple_four() {
    tuple_four::<Tuple>();
    //~^ ERROR not satisfied
}

fn main() { }
