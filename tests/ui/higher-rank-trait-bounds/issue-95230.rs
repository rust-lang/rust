// check-pass

pub struct Bar
where
    for<'a> &'a mut Self:;

fn main() {}
