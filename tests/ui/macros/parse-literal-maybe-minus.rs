// Tests some cases relating to `parse_literal_maybe_minus`, which used to
// accept more interpolated expressions than it should have.

const A: u32 = 0;
const B: u32 = 1;

macro_rules! identity {
    ($val:expr) => { $val }
}

macro_rules! range1 {
    ($min:expr) => {
        $min..
    }
}

macro_rules! range2 {
    ($min:expr, $max:expr) => {
        $min ..= $max
    }
}

macro_rules! range3 {
    ($max:expr) => {
        .. $max
    }
}

macro_rules! m {
    ($a:expr, $b:ident, $identity_mac_call:expr) => {
        fn _f() {
            match [1, 2] {
                [$a, $b] => {}  // FIXME: doesn't compile, probably should
                _ => {}
            }

            match (1, 2) {
                ($a, $b) => {}  // FIXME: doesn't compile, probably should
                _ => {}
            }

            match 3 {
                $identity_mac_call => {}
                0..$identity_mac_call => {}
                _ => {}
            }
        }
    };
}

m!(A, B, identity!(10));
//~^ ERROR arbitrary expressions aren't allowed in patterns
//~| ERROR arbitrary expressions aren't allowed in patterns

fn main() {
    match 3 {
        identity!(A) => {}  // FIXME: doesn't compile, probably should
        //~^ ERROR arbitrary expressions aren't allowed in patterns
        range1!(A) => {}
        range2!(A, B) => {}
        range3!(B) => {}
        _ => {}
    }
}
