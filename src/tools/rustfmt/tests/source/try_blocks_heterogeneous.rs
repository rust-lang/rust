// rustfmt-edition: 2018
#![feature(try_blocks_heterogeneous)]

fn main() -> Result<(), !> {
    let _x = try bikeshed Option<_> {
        4
    };

    try bikeshed Result<_, _> {}
}

fn baz() -> Option<i32> {
    if (1 == 1) {
        return try bikeshed Option<i32> {
            5
        };
    }

    // test
    let x = try bikeshed Option<()> {
        // try blocks are great
    };

    let y = try bikeshed Option<i32> {
        6
    }; // comment

    let x = try /* Invisible comment */ bikeshed Option<()> {};
    let x = try bikeshed /* Invisible comment */ Option<()> {};
    let x = try bikeshed Option<()> /* Invisible comment */ {};

    let x = try bikeshed Option<i32> { baz()?; baz()?; baz()?; 7 };

    let x = try bikeshed Foo<Bar, Bar, Bar, Bar, Bar, Bar, Bar, Bar, Bar, Bar, Bar, Bar, Bar> { 1 + 1 + 1 };

    let x = try bikeshed Foo<Bar, Bar, Bar, Bar, Bar, Bar, Bar, Bar, Bar, Bar, Bar, Bar, Bar, Bar> {};

    return None;
}
