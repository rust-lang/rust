// check-pass
// compile-flags: -Ztrait-solver=next
// Issue 96230

use std::fmt::Debug;

trait Classic {
    type Assoc;
}

trait Gat {
    type Assoc<'a>;
}

struct Foo;

impl Classic for Foo {
    type Assoc = ();
}

impl Gat for Foo {
    type Assoc<'i> = ();
}

fn classic_debug<T: Classic>(_: T)
where
    T::Assoc: Debug,
{
}

fn gat_debug<T: Gat>(_: T)
where
    for<'a> T::Assoc<'a>: Debug,
{
}

fn main() {
    classic_debug::<Foo>(Foo); // fine
    classic_debug(Foo); // fine

    gat_debug::<Foo>(Foo); // fine
    gat_debug(Foo); // boom
}
