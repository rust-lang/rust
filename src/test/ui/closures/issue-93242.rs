// check-pass
// edition:2021

pub fn something(path: &[usize]) -> impl Fn() -> usize + '_ {
    move || match path {
        [] => 0,
        _ => 1,
    }
}

fn main(){}
