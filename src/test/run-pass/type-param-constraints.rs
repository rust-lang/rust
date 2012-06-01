fn p_foo<T>(pinned: T) { }
fn s_foo<T: copy>(shared: T) { }
fn u_foo<T: send>(unique: T) { }

class r {
  let i: int;
  new(i:int) { self.i = i; }
  drop {}
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
