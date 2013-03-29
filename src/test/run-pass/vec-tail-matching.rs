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
            assert!(first.string == ~"foo");
            assert!(tail.len() == 2);
            assert!(tail[0].string == ~"bar");
            assert!(tail[1].string == ~"baz");

            match tail {
                [Foo { _ }, _, Foo { _ }, ..tail] => {
                    ::core::util::unreachable();
                }
                [Foo { string: a }, Foo { string: b }] => {
                    assert!(a == ~"bar");
                    assert!(b == ~"baz");
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
