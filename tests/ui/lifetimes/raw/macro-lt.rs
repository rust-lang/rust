//@ check-pass

macro_rules! lifetime {
    ($lt:lifetime) => {
        fn hello<$lt>() {}
    }
}

lifetime!('r#struct);

fn main() {}
