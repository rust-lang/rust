// This tests that exports can have visible dependencies on things
// that are not exported, allowing for a sort of poor-man's ADT

mod foo {
    export f;
    export g;

    // not exported
    enum t { t1, t2, }

    impl t : cmp::Eq {
        pure fn eq(&&other: t) -> bool {
            (self as uint) == (other as uint)
        }
        pure fn ne(&&other: t) -> bool { !self.eq(other) }
    }

    fn f() -> t { return t1; }

    fn g(v: t) { assert (v == t1); }
}

fn main() { foo::g(foo::f()); }
