// compile-flags: -W help
// check-pass
//
// ignore-tidy-linelength
//
// normalize-stdout-test: "( +name  default  meaning\n +----  -------  -------\n)?( *[[:word:]:-]+  (allow  |warn   |deny   |forbid )  [^\n]+\n)+" -> "    $$NAMES  $$LEVELS  $$MEANINGS"
// normalize-stdout-test: " +name  sub-lints\n +----  ---------\n( *[[:word:]:-]+  [^\n]+\n)+" -> "    $$NAMES  $$SUB_LINTS"
// normalize-stdout-test: " +rustdoc::all(  (rustdoc::[[:word:]-]+, )*rustdoc::[[:word:]-]+)?" -> "    rustdoc::all  $$GROUPS$4"
