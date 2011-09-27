
resource r(i: @mutable int) {
    *i += 1;
}

fn test1() {
    let i = @mutable 100;
    let j = @mutable 200;
    {
        let x <- ~r(i);
        let y <- ~r(j);
        x <-> y;
        assert ***x == 200;
        assert ***y == 100;
    }
    assert *i == 101;
    assert *j == 201;
}

fn test2() {
    let i = @mutable 0;
    {
        let x <- ~r(i);
        let y <- ~r(i);
        x <-> y;
    }
    assert *i == 2;
}

fn main() {
    test1();
    test2();
}