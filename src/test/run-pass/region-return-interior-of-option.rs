fn get<T>(opt: &r/Option<T>) -> &r/T {
    match *opt {
      Some(ref v) => v,
      None => fail ~"none"
    }
}

fn main() {
    let mut x = Some(23);

    {
        let y = get(&x);
        assert *y == 23;
    }

    x = Some(24);

    {
        let y = get(&x);
        assert *y == 24;
    }
}