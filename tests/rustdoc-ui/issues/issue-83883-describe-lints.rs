//@ compile-flags: -W help
//@ check-pass
//@ check-stdout
//
// ignore-tidy-linelength
//
//@ normalize-stdout: "( +name  default  meaning\n +----  -------  -------\n)?( *[[:word:]:-]+  (allow  |warn   |deny   |forbid )  [^\n]+\n)+" -> "    $$NAMES  $$LEVELS  $$MEANINGS"
//@ normalize-stdout: " +name  sub-lints\n +----  ---------\n( *[[:word:]:-]+  [^\n]+\n)+" -> "    $$NAMES  $$SUB_LINTS"

//~? RAW Lint checks provided
//~? RAW rustdoc::broken-intra-doc-links
