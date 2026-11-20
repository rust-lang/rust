// This test is illustrating the difference between how failing to derive
// `PartialEq` is handled compared to failing to implement it at all.

// See also RFC 1445

#[derive(PartialEq, Eq)]
struct Structural(u32);

struct NoPartialEq(u32);

struct NoDerive(u32);

// This impl makes NoDerive irreflexive.
impl PartialEq for NoDerive { fn eq(&self, _: &Self) -> bool { false } }

impl Eq for NoDerive { }

const NO_DERIVE_NONE: Option<NoDerive> = None;
const NO_PARTIAL_EQ_NONE: Option<NoPartialEq> = None;

fn main() {
    match None {
        NO_DERIVE_NONE => println!("NO_DERIVE_NONE"),
        _ => panic!("whoops"),
    }

    match None {
        NO_PARTIAL_EQ_NONE => println!("NO_PARTIAL_EQ_NONE"),
        //~^ ERROR constant of non-structural type `Option<NoPartialEq>` in a pattern
        _ => panic!("whoops"),
    }
}
