trait Id {
    type Out;

    fn id(self) -> Self::Out;
}

impl<'a> Id for &'a mut i32 {
    type Out = &'a mut i32;

    fn id(self) -> Self {
        self
    }
}

impl<'a> Id for &'a mut u32 {
    type Out = &'a mut u32;

    fn id(self) -> Self {
        self
    }
}

fn foo<T>(mut x: T)
where
    for<'a> &'a mut T: Id,
{
    let x = &mut x;
    let _y = x.id();
    // Inspecting the trace should show that `_y` has a type involving a local lifetime, when it gets validated.
    // Unfortunately, there doesn't seem to be a way to actually have a test fail if it does not have the right
    // type. Currently, this is *not* working correctly; see <https://github.com/rust-lang/miri/issues/298>.
}

fn main() {
    foo(3)
}
