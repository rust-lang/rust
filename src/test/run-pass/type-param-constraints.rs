fn p_foo<T>(pinned: T) { }
fn s_foo<T: copy>(shared: T) { }
fn u_foo<T: send>(unique: T) { }

struct r {
  let i: int;
  drop {}
}

fn r(i:int) -> r {
    r {
        i: i
    }
}

fn main() {
    p_foo(r(10));
    p_foo(@r(10));

    p_foo(~r(10));
    p_foo(@10);
    p_foo(~10);
    p_foo(10);

    s_foo(@r(10));
    s_foo(@10);
    s_foo(~10);
    s_foo(10);

    u_foo(~10);
    u_foo(10);
}
