// Test that we do not permit moves from &[] matched by a vec pattern.

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
            match tail {
                [Foo { string: a }, Foo { string: b }] => {
                    //~^ ERROR cannot move out of dereference of & pointer
                    //~^^ ERROR cannot move out of dereference of & pointer
                }
                _ => {
                    ::std::util::unreachable();
                }
            }
            let z = copy tail[0];
            debug!(fmt!("%?", z));
        }
        _ => {
            ::std::util::unreachable();
        }
    }
}
