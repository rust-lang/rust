// This originally crashed because `Recovery::Forbidden` wasn't being applied
// when fragments pasted by declarative macros were reparsed.

macro_rules! m {
    ($p:pat) => {
        if let $p = 0 {}
    }
}

fn main() {
    m!(0X0); //~ ERROR invalid base prefix for number literal
}
