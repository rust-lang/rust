// Bad `.clone()` suggestions

struct S<A> {
    t: A,
}

fn struct_field_shortcut_move<T: Clone>(t: T) {
    S { t };
    t; //~ use of moved value: `t`
}

fn closure_clone_will_not_help<T: Clone>(t: T) {
    (move || {
        t;
    })();
    t; //~ use of moved value: `t`
}

#[derive(Clone)]
struct CloneOnly;

fn update_syntax<T: Clone>(s: S<T>) {
    S { ..s };
    S { ..s }; //~ use of moved value: `s.t`
}

fn main() {}
