pub
impl(crate)
trait Foo {}

pub
impl
trait Bar {}

pub
impl ( in foo
::
bar )
trait Baz {}

struct FooS {
    pub
    mut(crate)
    field: (),
}

struct BarS {
    pub
    mut
    field: (),
}

struct BazS {
    pub
    mut ( in foo
    ::
    bar )
    field: (),
}

struct FooS2(
    pub
    mut(crate)
    (),
);

struct BarS2(
    pub
    mut
    (),
);

struct BazS2(
    pub
    mut ( in foo
    ::
    bar )
    (),
);

enum Enum {
    Foo {
        mut(crate)
        field: (),
    },
    Bar {
        mut
        field: (),
    },
    Baz {
        mut ( in foo
        ::
        bar )
        field: (),
    },
    FooT(
        mut(crate)
        (),
    ),
    BarT(
        mut
        (),
    ),
    BazT(
        mut ( in foo
        ::
        bar )
        (),
    ),

}

union Union {
    mut(crate)
    field1: (),
    mut
    field2: (),
    mut ( in foo
    ::
    bar )
    field3: (),
}
