// compile-flags: -Zno-analysis

type NonNullU32 = u32 is 1..;
//~^ pattern types are unstable
type Percent = u32 is 0..=100;
//~^ pattern types are unstable
type Negative = i32 is ..=0;
//~^ pattern types are unstable
type Positive = i32 is 0..;
//~^ pattern types are unstable
type Always = Option<u32> is Some(_);
//~^ pattern types are unstable
