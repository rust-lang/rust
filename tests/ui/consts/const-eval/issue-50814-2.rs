//@ build-fail
//@ revisions: normal mir-opt
//@ [mir-opt]compile-flags: -Zmir-opt-level=4
//@ dont-require-annotations: NOTE

trait C {
    const BOO: usize;
}

trait Foo<T> {
    const BAR: usize;
}

struct A<T>(T);

impl<T: C> Foo<T> for A<T> {
    const BAR: usize = [5, 6, 7][T::BOO]; //~ ERROR index out of bounds: the length is 3 but the index is 42
}

fn foo<T: C>() -> &'static usize {
    &<A<T> as Foo<T>>::BAR //~ NOTE constant
}

impl C for () {
    const BOO: usize = 42;
}

impl C for u32 {
    const BOO: usize = 1;
}

fn main() {
    println!("{:x}", foo::<()>() as *const usize as usize);
    println!("{:x}", foo::<u32>() as *const usize as usize);
    println!("{:x}", foo::<()>());
    println!("{:x}", foo::<u32>());
}
