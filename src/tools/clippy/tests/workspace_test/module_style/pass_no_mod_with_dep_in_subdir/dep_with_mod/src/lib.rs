pub mod with_mod;

pub fn foo() {
    let _ = with_mod::Thing;
    let _ = with_mod::inner::stuff::Inner;
    let _ = with_mod::inner::stuff::most::Snarks;
}
