// error-pattern:aFdEfSeVEE

/* We're testing that link_args are indeed passed when nolink is specified.
So we try to compile with junk link_args and make sure they are visible in
the compiler output. */

#[link_args = "aFdEfSeVEEE"]
#[nolink]
extern mod m1 {
    #[legacy_exports]; }

fn main() { }