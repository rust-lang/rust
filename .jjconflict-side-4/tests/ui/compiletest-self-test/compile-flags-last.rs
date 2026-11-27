// Check that the arguments provided through `// compile-flags` are added last to the command line
// in UI tests. To ensure that we invoke rustc with a flag that expects an argument withut actually
// providing it. If the compile-flags are not last, the test will fail as rustc will interpret the
// next flag as the argument of this flag.
//
//@ compile-flags: --cap-lints

//~? RAW Argument to option 'cap-lints' missing
