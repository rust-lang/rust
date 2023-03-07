pub trait Mirror<Smoke> {
    type Image;
}

impl<T, Smoke> Mirror<Smoke> for T {
    type Image = T;
}

pub fn poison<S>(victim: String) where <String as Mirror<S>>::Image: Copy {
    loop { drop(victim); }
}

fn main() {
    let s = "Hello!".to_owned();
    let mut s_copy = s;
    s_copy.push_str("World!");
    "0wned!".to_owned();
    println!("{}", s); //~ ERROR borrow of moved value
}
