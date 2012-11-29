use zoo::{duck, goose}; //~ ERROR failed to resolve import

mod zoo {
    pub enum bird {
        pub duck,
        priv goose
    }
}


fn main() {
    let y = goose;
}
