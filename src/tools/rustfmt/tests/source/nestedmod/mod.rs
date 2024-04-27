
mod mod2a;
mod mod2b;

mod mymod1 {
          use mod2a::{Foo,Bar};
mod mod3a;
}

#[path="mod2c.rs"]
mod mymod2;

mod submod2;
