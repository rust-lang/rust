struct Foo {
    string: ~str
}

pub fn main() {
    let x = [
        Foo { string: ~"foo" },
        Foo { string: ~"bar" },
        Foo { string: ~"baz" }
    ];
    match x {
        [first, ..tail] => {
            fail_unless!(first.string == ~"foo");
            fail_unless!(tail.len() == 2);
            fail_unless!(tail[0].string == ~"bar");
            fail_unless!(tail[1].string == ~"baz");

            match tail {
                [Foo { _ }, _, Foo { _ }, ..tail] => {
                    ::core::util::unreachable();
                }
                [Foo { string: a }, Foo { string: b }] => {
                    fail_unless!(a == ~"bar");
                    fail_unless!(b == ~"baz");
                }
                _ => {
                    ::core::util::unreachable();
                }
            }
        }
        _ => {
            ::core::util::unreachable();
        }
    }
}
