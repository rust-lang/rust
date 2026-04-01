// Check that deriving builtin traits for a packed struct with
// non-Copy fields emits move errors along with an additional
// diagnostic note explaining the reason
// See issue #117406

use std::fmt::{Debug, Formatter, Result};
use std::cmp::Ordering;

// Packed + derives: additional diagnostic should be emitted
// for each of Debug, PartialEq and PartialOrd
#[repr(packed)]
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Default)]
struct StructA(String);
//~^ ERROR: cannot move out of a shared reference [E0507]
//~| ERROR: cannot move out of a shared reference [E0507]
//~| ERROR: cannot move out of a shared reference [E0507]
//~| ERROR: cannot move out of a shared reference [E0507]
//~| ERROR: cannot move out of a shared reference [E0507]
//~| ERROR: cannot move out of a shared reference [E0507]
//~| ERROR: cannot move out of a shared reference [E0507]
//~| ERROR: cannot move out of a shared reference [E0507]
//~| ERROR: cannot move out of a shared reference [E0507]


// Unrelated impl: additinal diagnostic should NOT be emitted
impl StructA {
    fn fmt(&self) -> String {
        self.0 //~ ERROR: cannot move out of `self` which is behind a shared reference
    }
}

// Packed + manual impls: additional diagnostic should NOT be emitted
#[repr(packed)]
struct StructB(String);

impl Debug for StructB {
    fn fmt(&self, f: &mut Formatter) -> Result {
        let x = &{ self.0 }; //~ ERROR: cannot move out of `self` which is behind a shared reference
        write!(f, "{}", x)
    }
}

impl PartialEq for StructB {
    fn eq(&self, other: &StructB) -> bool {
        ({ self.0 }) == ({ other.0 })
        //~^ ERROR: cannot move out of `self` which is behind a shared reference
        //~| ERROR: cannot move out of `other` which is behind a shared reference
    }
}

impl PartialOrd for StructB {
    fn partial_cmp(&self, other: &StructB) -> Option<Ordering> {
        PartialOrd::partial_cmp(&{ self.0 }, &{ other.0 })
        //~^ ERROR: cannot move out of `self` which is behind a shared reference
        //~| ERROR: cannot move out of `other` which is behind a shared reference
    }
}

// NOT packed + derives: additinal diagnostic should NOT be emitted
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Default)]
struct StructC(String);

// NOT packed + manual impls: additinal dignostic should NOT be emitted
struct StructD(String);

impl Debug for StructD {
    fn fmt(&self, f: &mut Formatter) -> Result {
        let x = &{ self.0 }; //~ ERROR: cannot move out of `self` which is behind a shared reference
        write!(f, "{}", x)
    }
}

impl PartialEq for StructD {
    fn eq(&self, other: &StructD) -> bool {
        ({ self.0 }) == ({ other.0 })
        //~^ ERROR: cannot move out of `self` which is behind a shared reference
        //~| ERROR: cannot move out of `other` which is behind a shared reference
    }
}

impl PartialOrd for StructD {
    fn partial_cmp(&self, other: &StructD) -> Option<Ordering> {
        PartialOrd::partial_cmp(&{ self.0 }, &{ other.0 })
        //~^ ERROR: cannot move out of `self` which is behind a shared reference
        //~| ERROR: cannot move out of `other` which is behind a shared reference
    }
}

// Packed + derives but the move is outside of a derive
// expansion: additinal diagnostic should NOT be emitted
fn func(arg: &StructA) -> String {
    arg.0 //~ ERROR: cannot move out of `arg` which is behind a shared reference
}

fn main(){
}
