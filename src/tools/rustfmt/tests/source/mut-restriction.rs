#![feature(mut_restrictions, unsafe_fields)]

struct FooS {
    pub
    mut(crate)
    field1: (),
    pub
    mut(crate        
    )
    unsafe
    field2: (),
}

struct BazS {
    pub
    mut ( in foo
    ::
    bar )
    field1: (),
    pub
    mut(
        in foo
        ::
        bar
    ) unsafe
    field2: (),
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
        pub(crate) mut(self)
        field1: (),
        pub
        mut(self        
        )
        unsafe
        field2: (),
    },
    Baz {
        pub
        mut ( in foo
        ::
        bar )
        field1: (),
        pub(
            crate
        )
        mut(
            in foo
            ::
            bar
        ) unsafe field2: (),
    },
    FooT(
        pub(crate) 
        mut(self)
        (),
    ),
    BazT(
        pub(crate
        )
        mut ( in foo
        ::
        bar )
        (),
    ),

}

union Union {
    pub
    mut(crate)
    field1: (),
    pub(crate
    )
    mut ( in foo
    ::
    bar )
    field2: (),
    pub
    mut(crate        
    )
    unsafe
    field3: (),
    pub(
        crate
    )
    mut(
        in foo
        ::
        bar
    ) unsafe
    field4: (),
}
