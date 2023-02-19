fn check(list: &[Option<()>]) {
    match list {
    //~^ ERROR match is non-exhaustive [E0004]
        &[] => {},
        &[_] => {},
        &[_, _] => {},
        &[_, None, ..] => {},
        &[.., Some(_), _] => {},
    }
}

fn main() {}
