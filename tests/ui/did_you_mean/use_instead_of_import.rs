//@ run-rustfix

import std::{
    //~^ ERROR expected item, found `import`
    io::Write,
    rc::Rc,
};

require std::time::Duration;
//~^ ERROR expected item, found `require`

include std::time::Instant;
//~^ ERROR expected item, found `include`

pub using std::io;
//~^ ERROR expected item, found `using`

fn main() {
    let x = Rc::new(1);
    let _ = write!(io::stdout(), "{:?}", x);
    let _ = Duration::new(5, 0);
    let _ = Instant::now();
}
