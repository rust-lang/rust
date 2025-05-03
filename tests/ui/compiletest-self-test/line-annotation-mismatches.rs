//@ should-fail

// The warning is reported with unknown line
//@ compile-flags: -D raw_pointer_derive
//~? WARN kind and unknown line match the reported warning, but we do not suggest it

// The error is expected but not reported at all.
//~ ERROR this error does not exist

// The error is reported but not expected at all.
// "`main` function not found in crate" (the main function is intentionally not added)

// An "unimportant" diagnostic is expected on a wrong line.
//~ ERROR aborting due to

// An "unimportant" diagnostic is expected with a wrong kind.
//~? ERROR For more information about an error

fn wrong_line_or_kind() {
    // A diagnostic expected on a wrong line.
    unresolved1;
    //~ ERROR cannot find value `unresolved1` in this scope

    // A diagnostic expected with a wrong kind.
    unresolved2; //~ WARN cannot find value `unresolved2` in this scope

    // A diagnostic expected with a missing kind (treated as a wrong kind).
    unresolved3; //~ cannot find value `unresolved3` in this scope

    // A diagnostic expected with a wrong line and kind.
    unresolved4;
    //~ WARN cannot find value `unresolved4` in this scope
}

fn wrong_message() {
    // A diagnostic expected with a wrong message, but the line is known and right.
    unresolvedA; //~ ERROR stub message 1

    // A diagnostic expected with a wrong message, but the line is known and right,
    // even if the kind doesn't match.
    unresolvedB; //~ WARN stub message 2
}
