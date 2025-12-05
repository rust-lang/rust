//@ check-pass
//@ edition: 2021

macro_rules! lifetime {
    ($lt:lifetime) => {
        fn hello<$lt>() {}
    }
}

lifetime!('r#struct);

fn main() {}
