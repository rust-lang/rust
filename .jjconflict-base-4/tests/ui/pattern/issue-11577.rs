//@ run-pass
// Destructuring struct variants would ICE where regular structs wouldn't

enum Foo {
    VBar { num: isize }
}

struct SBar { num: isize }

pub fn main() {
    let vbar = Foo::VBar { num: 1 };
    let Foo::VBar { num } = vbar;
    assert_eq!(num, 1);

    let sbar = SBar { num: 2 };
    let SBar { num } = sbar;
    assert_eq!(num, 2);
}
