// Some non-controversial subset of ambiguities "modern macro name" vs "macro_rules"
// is disambiguated to mitigate regressions from macro modularization.
// Scoping for `macro_rules` behaves like scoping for `let` at module level, in general.

#![feature(decl_macro)]

fn same_unnamed_mod() {
    macro m() { 0 }

    macro_rules! m { () => (()) }

    m!() // OK
}

fn nested_unnamed_mod() {
    macro m() { 0 }

    {
        macro_rules! m { () => (()) }

        m!() // OK
    }
}

fn nested_unnamed_mod_fail() {
    macro_rules! m { () => (()) }

    {
        macro m() { 0 }

        m!() //~ ERROR `m` is ambiguous
    }
}

fn nexted_named_mod_fail() {
    macro m() { 0 }

    #[macro_use]
    mod inner {
        macro_rules! m { () => (()) }
    }

    m!() //~ ERROR `m` is ambiguous
}

fn main() {}
