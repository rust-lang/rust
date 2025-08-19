//@ check-fail

#[derive(Copy, Clone)]
pub struct ChildStdin {
    inner: AnonPipe,
}

#[derive(Copy, Clone)]
enum AnonPipe {}

const FOO: () = {
    union Foo {
        a: ChildStdin,
        b: (),
    }
    let x = unsafe { Foo { b: () }.a };
    //~^ ERROR: value of uninhabited type
    let x = &x.inner;
};

fn main() {}
