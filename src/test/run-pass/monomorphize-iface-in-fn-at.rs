// test that invoking functions which require
// dictionaries from inside an fn@ works
// (at one point, it didn't)

fn mk_nil<C:ty_ops>(cx: C) -> uint {
    cx.mk()
}

iface ty_ops {
    fn mk() -> uint;
}

impl of ty_ops for () {
    fn mk() -> uint { 22u }
}

fn main() {
    let fn_env = fn@() -> uint {
        mk_nil(())
    };
    assert fn_env() == 22u;
}