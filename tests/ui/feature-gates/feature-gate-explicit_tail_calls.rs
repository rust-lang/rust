pub fn you<T>() -> T {
    become bottom(); //~ error: `become` expression is experimental
}

pub fn bottom<T>() -> T {
    become you(); //~ error: `become` expression is experimental
}

fn main() {}
