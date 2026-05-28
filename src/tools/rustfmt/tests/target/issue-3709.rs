// rustfmt-edition: 2018

macro_rules! token {
    ($t:tt) => {};
}

fn main() {
    token!(dyn);
    token!(dyn);
}
