// Resgression test for #107516.
//@ check-pass

fn iter1<'a: 'a>() -> impl Iterator<Item = &'static str> {
    None.into_iter()
}

fn iter2<'a>() -> impl Iterator<Item = &'a str> {
    None.into_iter()
}

struct Bivar<'a, I: Iterator<Item = &'a str> + 'a>(I);

fn main() {
    let _ = || Bivar(iter1());
    let _ = || Bivar(iter2());
}
