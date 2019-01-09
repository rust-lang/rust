// Ensure that a user-defined type admits multiple inherent methods
// with the same name, which can be called on values that have a
// precise enough type to allow distinguishing between the methods.


struct Foo<T>(T);

impl Foo<usize> {
    fn bar(&self) -> i32 { self.0 as i32 }
}

impl Foo<isize> {
    fn bar(&self) -> i32 { -(self.0 as i32) }
}

fn main() {
    let foo_u = Foo::<usize>(5);
    assert_eq!(foo_u.bar(), 5);

    let foo_i = Foo::<isize>(3);
    assert_eq!(foo_i.bar(), -3);
}
