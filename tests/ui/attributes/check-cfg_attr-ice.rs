//! I missed a `cfg_attr` match in #128581, it should have had the same treatment as `cfg`. If
//! an invalid attribute starting with `cfg_attr` is passed, then it would trigger an ICE because
//! it was not considered "checked" (e.g. `#[cfg_attr::skip]` or `#[cfg_attr::no_such_thing]`).
//!
//! This test is not exhaustive, there's too many possible positions to check, instead it just does
//! a basic smoke test in a few select positions to make sure we don't ICE for e.g.
//! `#[cfg_attr::no_such_thing]`.
//!
//! issue: rust-lang/rust#128716
#![crate_type = "lib"]

#[cfg_attr::no_such_thing]
//~^ ERROR failed to resolve
mod we_are_no_strangers_to_love {}

#[cfg_attr::no_such_thing]
//~^ ERROR failed to resolve
struct YouKnowTheRules {
    #[cfg_attr::no_such_thing]
    //~^ ERROR failed to resolve
    and_so_do_i: u8,
}

#[cfg_attr::no_such_thing]
//~^ ERROR failed to resolve
fn a_full_commitment() {
    #[cfg_attr::no_such_thing]
    //~^ ERROR failed to resolve
    let is_what_i_am_thinking_of = ();
}

#[cfg_attr::no_such_thing]
//~^ ERROR failed to resolve
union AnyOtherGuy {
    owo: ()
}
struct This;

#[cfg_attr(FALSE, doc = "you wouldn't get this")]
impl From<AnyOtherGuy> for This {
    #[cfg_attr::no_such_thing]
    //~^ ERROR failed to resolve
    fn from(#[cfg_attr::no_such_thing] any_other_guy: AnyOtherGuy) -> This {
        //~^ ERROR failed to resolve
        #[cfg_attr::no_such_thing]
        //~^ ERROR attributes on expressions are experimental
        //~| ERROR failed to resolve
        unreachable!()
    }
}

#[cfg_attr::no_such_thing]
//~^ ERROR failed to resolve
enum NeverGonna {
    #[cfg_attr::no_such_thing]
    //~^ ERROR failed to resolve
    GiveYouUp(#[cfg_attr::no_such_thing] u8),
    //~^ ERROR failed to resolve
    LetYouDown {
        #![cfg_attr::no_such_thing]
        //~^ ERROR an inner attribute is not permitted in this context
        never_gonna: (),
        round_around: (),
        #[cfg_attr::no_such_thing]
        //~^ ERROR failed to resolve
        and_desert_you: (),
    },
}
