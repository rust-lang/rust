pub trait Trait {
    fn fun(par: &str) -> &str;
}

impl Trait for str {
    fn fun(par: &str) -> &str {
        &par[0..1]
    }
}
