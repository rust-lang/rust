fn get<T>(opt: &r/option<T>) -> &r/T {
    match *opt {
      some(ref v) => v,
      none => fail ~"none"
    }
}

fn main() {
    let mut x = some(23);

    {
        let y = get(&x);
        assert *y == 23;
    }

    x = some(24);

    {
        let y = get(&x);
        assert *y == 24;
    }
}