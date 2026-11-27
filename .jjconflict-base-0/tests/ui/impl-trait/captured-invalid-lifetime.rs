// This uses edition 2024 for new lifetime capture rules.
//@ edition: 2024

// The problem here is that the presence of the opaque which captures all lifetimes in scope
// means that the duplicated `'a` (which I'll call the dupe) is considered to be *early-bound*
// since it shows up in the output but not the inputs. This is paired with the fact that we
// were previously setting the name of the dupe to `'_` in the generic param definition, which
// means that the identity args for the function were `['a#0, '_#1]` even though the lifetime
// for the dupe should've been `'a#1`. This difference in symbol meant that NLL couldn't
// actually match the lifetime against the identity lifetimes, leading to an ICE.

struct Foo<'a>(&'a ());

impl<'a> Foo<'a> {
    fn pass<'a>() -> impl Sized {}
    //~^ ERROR lifetime name `'a` shadows a lifetime name that is already in scope
}

fn main() {}
