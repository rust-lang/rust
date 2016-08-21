//  When testing functions, QuickCheck (QC) uses small values for integer (`u*`/`i*`) arguments
// (~ `[-100, 100]`), but these values don't stress all the code paths in our intrinsics. Here we
// create newtypes over the primitive integer types with the goal of having full control over the
// random values that will be used to test our intrinsics.

use std::boxed::Box;
use std::fmt;

use quickcheck::{Arbitrary, Gen};

use int::LargeInt;

// Generates values in the full range of the integer type
macro_rules! arbitrary {
    ($TY:ident : $ty:ident) => {
        #[derive(Clone, Copy)]
        pub struct $TY(pub $ty);

        impl Arbitrary for $TY {
            fn arbitrary<G>(g: &mut G) -> $TY
                where G: Gen
            {
                // NOTE Generate edge cases with a 10% chance
                let t = if g.gen_weighted_bool(10) {
                    *g.choose(&[
                        $ty::min_value(),
                        0,
                        $ty::max_value(),
                    ]).unwrap()
                } else {
                    g.gen()
                };

                $TY(t)
            }

            fn shrink(&self) -> Box<Iterator<Item=$TY>> {
                struct Shrinker {
                    x: $ty,
                }

                impl Iterator for Shrinker {
                    type Item = $TY;

                    fn next(&mut self) -> Option<$TY> {
                        self.x /= 2;
                        if self.x == 0 {
                            None
                        } else {
                            Some($TY(self.x))
                        }
                    }
                }

                if self.0 == 0 {
                    ::quickcheck::empty_shrinker()
                } else {
                    Box::new(Shrinker { x: self.0 })
                }
            }
        }

        impl fmt::Debug for $TY {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                fmt::Debug::fmt(&self.0, f)
            }
        }
    }
}

arbitrary!(I32: i32);
arbitrary!(U32: u32);

// These integers are "too large". If we generate e.g. `u64` values in the full range then there's
// only `1 / 2^32` chance of seeing a value smaller than `2^32` (i.e. whose higher "word" (32-bits)
// is `0`)! But this is an important group of values to tests because we have special code paths for
// them. Instead we'll generate e.g. `u64` integers this way: uniformly pick between (a) setting the
// low word to 0 and generating a random high word, (b) vice versa: high word to 0 and random low
// word or (c) generate both words randomly. This let's cover better the code paths in our
// intrinsics.
macro_rules! arbitrary_large {
    ($TY:ident : $ty:ident) => {
        #[derive(Clone, Copy)]
        pub struct $TY(pub $ty);

        impl Arbitrary for $TY {
            fn arbitrary<G>(g: &mut G) -> $TY
                where G: Gen
            {
                // NOTE Generate edge cases with a 10% chance
                let t = if g.gen_weighted_bool(10) {
                    *g.choose(&[
                        $ty::min_value(),
                        0,
                        $ty::max_value(),
                    ]).unwrap()
                } else {
                    match g.gen_range(0, 3) {
                        0 => $ty::from_parts(g.gen(), g.gen()),
                        1 => $ty::from_parts(0, g.gen()),
                        2 => $ty::from_parts(g.gen(), 0),
                        _ => unreachable!(),
                    }
                };

                $TY(t)
            }

            fn shrink(&self) -> Box<Iterator<Item=$TY>> {
                struct Shrinker {
                    x: $ty,
                }

                impl Iterator for Shrinker {
                    type Item = $TY;

                    fn next(&mut self) -> Option<$TY> {
                        self.x /= 2;
                        if self.x == 0 {
                            None
                        } else {
                            Some($TY(self.x))
                        }
                    }
                }

                if self.0 == 0 {
                    ::quickcheck::empty_shrinker()
                } else {
                    Box::new(Shrinker { x: self.0 })
                }
            }
        }

        impl fmt::Debug for $TY {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                fmt::Debug::fmt(&self.0, f)
            }
        }
    }
}

arbitrary_large!(I64: i64);
arbitrary_large!(U64: u64);
