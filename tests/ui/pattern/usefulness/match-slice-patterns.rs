fn check(list: &[Option<()>]) {
    match list {
    //~^ ERROR `&[_, Some(_), .., None, _]` not covered
        &[] => {},
        &[_] => {},
        &[_, _] => {},
        &[_, None, ..] => {},
        &[.., Some(_), _] => {},
    }
}

fn main() {}
