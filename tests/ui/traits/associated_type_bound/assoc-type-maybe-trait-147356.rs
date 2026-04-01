use std::str::FromStr;
fn foo<T: FromStr>() -> T
where
    <T as FromStr>::Err: Debug, //~ ERROR expected trait
{
    "".parse().unwrap()
}

fn bar<T: FromStr>() -> T
where
    <T as FromStr>::Err: some_unknown_name, //~ ERROR cannot find trait `some_unknown_name` in this scope
{
    "".parse().unwrap()
}

fn main() {}
