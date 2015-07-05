
mod mod2a;
mod mod2b;

mod mymod1 {
          use mod2a::{Foo,Bar};
}

#[path="mod2c.rs"]
mod mymod2;

mod submod2;
