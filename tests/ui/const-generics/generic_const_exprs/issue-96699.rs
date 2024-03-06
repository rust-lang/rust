//@ check-pass

#![allow(dead_code, incomplete_features)]
#![feature(generic_const_exprs)]

const fn min(a: usize, b: usize) -> usize {
    if a < b {
        a
    } else {
        b
    }
}

trait Trait1<Inner1>
where
    Self: Sized,
{
    fn crash_here()
    where
        Inner1: Default,
    {
        Inner1::default();
    }
}

struct Struct1<T>(T);
impl<T> Trait1<T> for Struct1<T> {}

trait Trait2<Inner2>
where
    Self: Sized,
{
    type Assoc: Trait1<Inner2>;

    fn call_crash()
    where
        Inner2: Default,
    {
        // if Inner2 implements Default, we can call crash_here.
        Self::Assoc::crash_here();
    }
}

struct Struct2<const SIZE1: usize, const SIZE2: usize> {}
/*
where
    [(); min(SIZE1, SIZE2)]:,
{
    elem: [i32; min(SIZE1, SIZE2)],
}
*/

impl<const SIZE1: usize, const SIZE2: usize> Trait2<[i32; min(SIZE1, SIZE2)]>
    for Struct2<SIZE1, SIZE2>
{
    type Assoc = Struct1<[i32; min(SIZE1, SIZE2)]>;
    // dose Struct1<[i32; min(SIZE1, SIZE2)]> implement Default?
}

fn main() {
    pattern2();

    print_fully_name(<Struct2<1, 2> as Trait2<[i32; min(1, 2)]>>::Assoc::crash_here);
    // <compiler_bug2::Struct1<[i32; 1]> as compiler_bug2::Trait1<[i32; 1]>>::crash_here
}

fn pattern1() {
    // no crash
    <Struct2<1, 2> as Trait2<[i32; min(1, 2)]>>::Assoc::crash_here();
    <Struct2<1, 2> as Trait2<[i32; min(1, 2)]>>::call_crash();
}

fn pattern2() {
    // crash
    <Struct2<1, 2> as Trait2<[i32; min(1, 2)]>>::call_crash();

    // undefined reference to `compiler_bug2::Trait1::crash_here'
}

fn pattern3() {
    // no crash
    <Struct2<1, 2> as Trait2<[i32; min(1, 2)]>>::Assoc::crash_here();
}

fn print_fully_name<T>(_: T) {
    let _ = std::any::type_name::<T>();
}
