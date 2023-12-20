// check-fail

macro_rules! macro_rules { () => {} }
//~^ ERROR: user-defined macros may not be named `macro_rules`

macro_rules! {}
//~^ ERROR: expected identifier, found `{`
//~| HELP: maybe you have forgotten to define a name for this `macro_rules!`
