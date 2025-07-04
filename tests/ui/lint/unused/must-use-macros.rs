// Makes sure the suggestions of the `unused_must_use` lint are not inside
//
// See <https://github.com/rust-lang/rust/issues/143025>

//@ check-pass
//@ run-rustfix

#![expect(unused_macros)]
#![warn(unused_must_use)]

fn main() {
    {
        macro_rules! cmp {
            ($a:tt, $b:tt) => {
                $a == $b
            };
        }

        // FIXME(Urgau): For some unknown reason the spans we get are not
        // recorded to be from any expansions, preventing us from either
        // suggesting in front of the macro or not at all.
        // cmp!(1, 1);
    }

    {
        macro_rules! cmp {
            ($a:ident, $b:ident) => {
                $a == $b
            }; //~^ WARN unused comparison that must be used
        }

        let a = 1;
        let b = 1;
        cmp!(a, b);
        //~^ SUGGESTION let _
    }

    {
        macro_rules! cmp {
            ($a:expr, $b:expr) => {
                $a == $b
            }; //~^ WARN unused comparison that must be used
        }

        cmp!(1, 1);
        //~^ SUGGESTION let _
    }

    {
        macro_rules! cmp {
            ($a:tt, $b:tt) => {
                $a.eq(&$b)
            };
        }

        cmp!(1, 1);
        //~^ WARN unused return value
        //~| SUGGESTION let _
    }
}
