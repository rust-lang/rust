// run-rustfix

import std::{
    //~^ ERROR expected item, found `import`
    io::Write,
    rc::Rc,
};

pub using std::io;
//~^ ERROR expected item, found `using`

fn main() {
    let x = Rc::new(1);
    let _ = write!(io::stdout(), "{:?}", x);
}
