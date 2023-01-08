pub
impl(crate)
trait Foo {}

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
    mut ( in foo
    ::
    bar )
    field3: (),
}
