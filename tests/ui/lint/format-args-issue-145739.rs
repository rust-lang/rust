use std::cell::Cell;
use std::fmt::{self, Display, Formatter};

#[derive(Debug)]
struct NonFreeze(Cell<usize>);

impl Display for DropImpl {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "DropImpl")
    }
}

#[derive(Debug)]
struct DropImpl;

impl Drop for DropImpl {
    fn drop(&mut self) {}
}

const NON_FREEZE: NonFreeze = NonFreeze(Cell::new(0));
const DROP_IMPL: DropImpl = DropImpl;
const FINE: &str = "fine";

fn main() {
    // These are okay
    format_args!("{FINE}{FINE}");
    format_args!("{DROP_IMPL}{DropImpl}");
    let _ = format!("{NON_FREEZE:?}{:?}", NON_FREEZE);

    // These are not
    print!("{DROP_IMPL}, {DROP_IMPL}");
    //~^ ERROR `format_args!` is called with multiple parameters that capturing the same constant violating issue 145739 [issue_145739]
    println!("{DropImpl}{DropImpl:?}");
    //~^ ERROR `format_args!` is called with multiple parameters that capturing the same constant constructor violating issue 145739 [issue_145739]
    format!("{NON_FREEZE:?}{:?}{NON_FREEZE:?}", NON_FREEZE);
    //~^ ERROR `format_args!` is called with multiple parameters that capturing the same constant violating issue 145739 [issue_145739]
    format!("{NON_FREEZE:?}{DropImpl}{NON_FREEZE:?}{DropImpl}");
    //~^ ERROR `format_args!` is called with multiple parameters that capturing the same constant violating issue 145739 [issue_145739]
    //~| ERROR `format_args!` is called with multiple parameters that capturing the same constant constructor violating issue 145739 [issue_145739]
}
