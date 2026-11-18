macro_rules! wat {
    ($($tt:tt)*) => {}
}

fn main(){
    #[deny(keyword_idents_2024)]
    //~^ WARN the `deny` attribute cannot be used on macro calls
    wat!(gen);
    //~^ ERROR `gen` is a keyword in the 2024 edition
    //~| WARN this is accepted in the current edition (Rust 2015) but is a hard error in Rust 2024!
}
