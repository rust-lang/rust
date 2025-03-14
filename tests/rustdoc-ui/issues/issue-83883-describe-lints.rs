//@ compile-flags: -W help
//@ check-pass
//@ check-stdout
//@ error-pattern:Lint checks provided
//@ error-pattern:rustdoc::broken-intra-doc-links
//
// ignore-tidy-linelength
//
//@ normalize-stdout: "( +name  default  meaning\n +----  -------  -------\n)?( *[[:word:]:-]+  (allow  |warn   |deny   |forbid )  [^\n]+\n)+" -> "    $$NAMES  $$LEVELS  $$MEANINGS"
//@ normalize-stdout: " +name  sub-lints\n +----  ---------\n( *[[:word:]:-]+  [^\n]+\n)+" -> "    $$NAMES  $$SUB_LINTS"
