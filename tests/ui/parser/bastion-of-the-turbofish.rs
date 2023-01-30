// check-pass

// Bastion of the Turbofish
// ------------------------
// Beware travellers, lest you venture into waters callous and unforgiving,
// where hope must be abandoned, ere it is cruelly torn from you. For here
// stands the bastion of the Turbofish: an impenetrable fortress holding
// unshaking against those who would dare suggest the supererogation of the
// Turbofish.
//
// Once I was young and foolish and had the impudence to imagine that I could
// shake free from the coils by which that creature had us tightly bound. I
// dared to suggest that there was a better way: a brighter future, in which
// Rustaceans both new and old could be rid of that vile beast. But alas! In
// my foolhardiness my ignorance was unveiled and my dreams were dashed
// unforgivingly against the rock of syntactic ambiguity.
//
// This humble program, small and insignificant though it might seem,
// demonstrates that to which we had previously cast a blind eye: an ambiguity
// in permitting generic arguments to be provided without the consent of the
// Great Turbofish. Should you be so naïve as to try to revolt against its
// mighty clutches, here shall its wrath be indomitably displayed. This
// program must pass for all eternity: forever watched by the guardian angel
// which gave this beast its name, and stands fundamentally at odds with the
// impetuous rebellion against the Turbofish.
//
// My heart aches in sorrow, for I know I am defeated. Let this be a warning
// to all those who come after: for they too must overcome the impassible
// hurdle of defeating the great beast, championed by a resolute winged
// guardian.
//
// Here stands the Bastion of the Turbofish, a memorial to Anna Harren,
// Guardian Angel of these Hallowed Grounds. <3

// See https://github.com/rust-lang/rust/pull/53562
// and https://github.com/rust-lang/rfcs/pull/2527
// and https://web.archive.org/web/20211010063452/https://twitter.com/garblefart/status/1393236602856611843
// for context.

fn main() {
    let (the, guardian, stands, resolute) = ("the", "Turbofish", "remains", "undefeated");
    let _: (bool, bool) = (the<guardian, stands>(resolute));
}
