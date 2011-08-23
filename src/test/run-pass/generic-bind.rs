

fn id<@T>(t: &T) -> T { ret t; }

fn main() {
    let t = {_0: 1, _1: 2, _2: 3, _3: 4, _4: 5, _5: 6, _6: 7};
    assert (t._5 == 6);
    let f1 =
        bind id::<{_0: int,
                   _1: int,
                   _2: int,
                   _3: int,
                   _4: int,
                   _5: int,
                   _6: int}>(_);
    assert (f1(t)._5 == 6);
}
