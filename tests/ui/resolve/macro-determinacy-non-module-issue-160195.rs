//@ revisions: env_first env_second

#[cfg(env_first)]
pub mod env {
    #[derive(Default)]
    pub struct BusinessData;
}

pub mod interface {
    use crate::env::{self};

    include!(concat!(env!())); //~ ERROR `env!()` takes 1 or 2 arguments
}

#[cfg(env_second)]
pub mod env {
    #[derive(Default)]
    pub struct BusinessData;
}

fn main() {}
