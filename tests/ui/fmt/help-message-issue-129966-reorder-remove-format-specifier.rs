// Test checks for the help messages in the error output/
//@ check-run-results

struct Foo(u8, u8);

fn main() {
    let f = Foo(1, 2);
    format!("{f:?#}");
//~^ ERROR invalid format string: unknown format identifier '?#'
    format!("{f:?x}");
//~^ ERROR invalid format string: unknown format identifier '?x'
    format!("{f:?X}");
//~^ ERROR invalid format string: unknown format identifier '?X'
}
