#![allow(incomplete_features)]
#![feature(unnamed_fields)]

#[derive(Clone, Copy)]
#[repr(C)]
struct Foo {
    a: u8,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct Bar {
    _: union {
        a: u8,
    },
}


// duplicated with a normal field
#[derive(Clone, Copy)]
#[repr(C)]
union A {
    // referent field
    a: u8,

    // normal field
    a: u8, //~ ERROR field `a` is already declared [E0124]
    // nested field
    _: struct {
        a: u8, //~ ERROR field `a` is already declared [E0124]
        a: u8, //~ ERROR field `a` is already declared [E0124]
    },
    // more nested field
    _: union {
        _: struct {
            a: u8, //~ ERROR field `a` is already declared [E0124]
        },
    },
    // nested field in a named adt
    _: Foo, //~ ERROR field `a` is already declared
    _: Bar, //~ ERROR field `a` is already declared
    // nested field in a named adt in an anoymous adt
    _: struct {
        _: Foo, //~ ERROR field `a` is already declared
        _: Bar, //~ ERROR field `a` is already declared
    },
}

// duplicated with a nested field
#[derive(Clone, Copy)]
#[repr(C)]
struct B {
    _: union {
        // referent field
        a: u8,

        // normal field (within the same anonymous adt)
        a: u8, //~ ERROR field `a` is already declared [E0124]
        // nested field (within the same anonymous adt)
        _: struct {
            a: u8, //~ ERROR field `a` is already declared [E0124]
        },
        // more nested field (within the same anonymous adt)
        _: union {
            _: struct {
                a: u8, //~ ERROR field `a` is already declared [E0124]
            },
        },
        // nested field in a named adt (within the same anonymous adt)
        _: Foo, //~ ERROR field `a` is already declared
        _: Bar, //~ ERROR field `a` is already declared
        // nested field in a named adt in an anoymous adt (within the same anonymous adt)
        _: struct {
            _: Foo, //~ ERROR field `a` is already declared
            _: Bar, //~ ERROR field `a` is already declared
        },
    },

    // normal field
    a: u8, //~ ERROR field `a` is already declared [E0124]
    // nested field
    _: struct {
        a: u8, //~ ERROR field `a` is already declared [E0124]
    },
    // more nested field
    _: union {
        _: struct {
            a: u8, //~ ERROR field `a` is already declared [E0124]
        },
    },
    // nested field in a named adt
    _: Foo, //~ ERROR field `a` is already declared
    _: Bar, //~ ERROR field `a` is already declared
    // nested field in a named adt in an anoymous adt
    _: struct {
        _: Foo, //~ ERROR field `a` is already declared
        _: Bar, //~ ERROR field `a` is already declared
    },
}

// duplicated with a more nested field
#[derive(Clone, Copy)]
#[repr(C)]
union C {
    _: struct {
        _: union {
            // referent field
            a: u8,

            // normal field (within the same anonymous adt)
            a: u8, //~ ERROR field `a` is already declared [E0124]
            // nested field (within the same anonymous adt)
            _: struct {
                a: u8, //~ ERROR field `a` is already declared [E0124]
            },
            // more nested field (within the same anonymous adt)
            _: union {
                _: struct {
                    a: u8, //~ ERROR field `a` is already declared [E0124]
                },
            },
            // nested field in a named adt (within the same anonymous adt)
            _: Foo, //~ ERROR field `a` is already declared
            _: Bar, //~ ERROR field `a` is already declared
            // nested field in a named adt in an anoymous adt (within the same anonymous adt)
            _: struct {
                _: Foo, //~ ERROR field `a` is already declared
                _: Bar, //~ ERROR field `a` is already declared
            },
        },

        // normal field (within the direct outer anonymous adt)
        a: u8, //~ ERROR field `a` is already declared [E0124]
        // nested field (within the direct outer anonymous adt)
        _: struct {
            a: u8, //~ ERROR field `a` is already declared [E0124]
        },
        // more nested field (within the direct outer anonymous adt)
        _: union {
            _: struct {
                a: u8, //~ ERROR field `a` is already declared [E0124]
            },
        },
        // nested field in a named adt (within the direct outer anonymous adt)
        _: Foo, //~ ERROR field `a` is already declared
        _: Bar, //~ ERROR field `a` is already declared
        // nested field in a named adt in an anoymous adt (within the direct outer anonymous adt)
        _: struct {
            _: Foo, //~ ERROR field `a` is already declared
            _: Bar, //~ ERROR field `a` is already declared
        },
    },
    // normal field
    a: u8, //~ ERROR field `a` is already declared [E0124]
    // nested field
    _: union {
        a: u8, //~ ERROR field `a` is already declared [E0124]
    },
    // more nested field
    _: struct {
        _: union {
            a: u8, //~ ERROR field `a` is already declared [E0124]
        },
    },
    // nested field in a named adt
    _: Foo, //~ ERROR field `a` is already declared
    _: Bar, //~ ERROR field `a` is already declared
    // nested field in a named adt in an anoymous adt
    _: union {
        _: Foo, //~ ERROR field `a` is already declared
        _: Bar, //~ ERROR field `a` is already declared
    },
}

// duplicated with a nested field in a named adt
#[derive(Clone, Copy)]
#[repr(C)]
struct D {
    // referent field `a`
    _: Foo,

    // normal field
    a: u8, //~ ERROR field `a` is already declared
    // nested field
    _: union {
        a: u8, //~ ERROR field `a` is already declared
    },
    // more nested field
    _: struct {
        _: union {
            a: u8, //~ ERROR field `a` is already declared
        },
    },
    // nested field in another named adt
    _: Foo, //~ ERROR field `a` is already declared
    _: Bar, //~ ERROR field `a` is already declared
    // nested field in a named adt in an anoymous adt
    _: union {
        _: Foo, //~ ERROR field `a` is already declared
        _: Bar, //~ ERROR field `a` is already declared
    },
}

// duplicated with a nested field in a nested field of a named adt
#[derive(Clone, Copy)]
#[repr(C)]
union D2 {
    // referent field `a`
    _: Bar,

    // normal field
    a: u8, //~ ERROR field `a` is already declared
    // nested field
    _: union {
        a: u8, //~ ERROR field `a` is already declared
    },
    // more nested field
    _: struct {
        _: union {
            a: u8, //~ ERROR field `a` is already declared
        },
    },
    // nested field in another named adt
    _: Foo, //~ ERROR field `a` is already declared
    _: Bar, //~ ERROR field `a` is already declared
    // nested field in a named adt in an anoymous adt
    _: union {
        _: Foo, //~ ERROR field `a` is already declared
        _: Bar, //~ ERROR field `a` is already declared
    },
}

// duplicated with a nested field in a named adt in an anonymous adt
#[derive(Clone, Copy)]
#[repr(C)]
struct E {
    _: struct {
        // referent field `a`
        _: Foo,

        // normal field (within the same anonymous adt)
        a: u8, //~ ERROR field `a` is already declared
        // nested field (within the same anonymous adt)
        _: struct {
            a: u8, //~ ERROR field `a` is already declared
        },
        // more nested field (within the same anonymous adt)
        _: union {
            _: struct {
                a: u8, //~ ERROR field `a` is already declared
            },
        },
        // nested field in a named adt (within the same anonymous adt)
        _: Foo, //~ ERROR field `a` is already declared
        _: Bar, //~ ERROR field `a` is already declared
        // nested field in a named adt in an anoymous adt (within the same anonymous adt)
        _: struct {
            _: Foo, //~ ERROR field `a` is already declared
            _: Bar, //~ ERROR field `a` is already declared
        },
    },

    // normal field
    a: u8, //~ ERROR field `a` is already declared
    // nested field
    _: union {
        a: u8, //~ ERROR field `a` is already declared
    },
    // more nested field
    _: struct {
        _: union {
            a: u8, //~ ERROR field `a` is already declared
        },
    },
    // nested field in another named adt
    _: Foo, //~ ERROR field `a` is already declared
    _: Bar, //~ ERROR field `a` is already declared
    // nested field in a named adt in an anoymous adt
    _: union {
        _: Foo, //~ ERROR field `a` is already declared
        _: Bar, //~ ERROR field `a` is already declared
    },
}

// duplicated with a nested field in a named adt in an anonymous adt
#[repr(C)]
#[derive(Clone, Copy)]
union E2 {
    _: struct {
        // referent field `a`
        _: Bar,

        // normal field (within the same anonymous adt)
        a: u8, //~ ERROR field `a` is already declared
        // nested field (within the same anonymous adt)
        _: struct {
            a: u8, //~ ERROR field `a` is already declared
        },
        // more nested field (within the same anonymous adt)
        _: union {
            _: struct {
                a: u8, //~ ERROR field `a` is already declared
            },
        },
        // nested field in a named adt (within the same anonymous adt)
        _: Foo, //~ ERROR field `a` is already declared
        _: Bar, //~ ERROR field `a` is already declared
        // nested field in a named adt in an anoymous adt (within the same anonymous adt)
        _: struct {
            _: Foo, //~ ERROR field `a` is already declared
            _: Bar, //~ ERROR field `a` is already declared
        },
    },

    // normal field
    a: u8, //~ ERROR field `a` is already declared
    // nested field
    _: union {
        a: u8, //~ ERROR field `a` is already declared
    },
    // more nested field
    _: struct {
        _: union {
            a: u8, //~ ERROR field `a` is already declared
        },
    },
    // nested field in another named adt
    _: Foo, //~ ERROR field `a` is already declared
    _: Bar, //~ ERROR field `a` is already declared
    // nested field in a named adt in an anoymous adt
    _: union {
        _: Foo, //~ ERROR field `a` is already declared
        _: Bar, //~ ERROR field `a` is already declared
    },
}

fn main() {}
