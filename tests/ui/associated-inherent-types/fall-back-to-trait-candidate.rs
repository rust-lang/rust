// Ensure that IAT selection doesn't hard error on associated type paths that could refer to
// an inherent associated type if we can't find an applicable inherent candidate since there
// might still be valid trait associated type candidates.
//
// FIXME(#142006): This only covers the bare minimum, we also need to disqualify inherent
// candidates if they're inaccessible or if the impl headers don't match / apply.
//
// issue: <https://github.com/rust-lang/rust/issues/142006#issuecomment-2938846613>
//@ check-pass

#![feature(inherent_associated_types)]
#![expect(incomplete_features)]

struct Type;
trait Trait { type AssocTy; fn scope(); }

impl Trait for Type {
    type AssocTy = ();

    fn scope() {
        let (): Self::AssocTy;
    }
}

fn main() { <Type as Trait>::scope(); }
