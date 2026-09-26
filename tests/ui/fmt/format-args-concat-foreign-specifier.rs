// When the format string comes from `concat!`, the offsets of printf- and shell-style specifiers
// are relative to the expanded string, not to the source. Before the fix, `check_foreign!` used
// them to place a machine-applicable suggestion at a bogus location in the `concat!` invocation,
// and rendering it caused an ICE if that location was inside a multibyte character. After the
// fix, the translation is explained in a note instead of a suggestion.

fn main() {
    // Used to ICE: the specifier's span lands inside `é` (2 bytes) or `𐏿` (4 bytes).
    format_args!(concat!("é", "aaaaa%d"), 1);
    //~^ ERROR argument never used
    format_args!(concat!("𐏿", "aaa%d"), 1);
    //~^ ERROR argument never used
    format_args!(concat!("é", "aaaaa$1"), 1);
    //~^ ERROR argument never used
    format_args!(concat!("𐏿", "aaa$1"), 1);
    //~^ ERROR argument never used

    // Used to suggest rewriting `concat!` itself, e.g. into `c{}cat!`.
    format_args!(concat!("%d", ""), 1);
    //~^ ERROR argument never used
    format_args!(concat!("$1", ""), 1);
    //~^ ERROR argument never used

    // Several specifiers each get their own note.
    format_args!(concat!("%d %s", ""), 1, 2);
    //~^ ERROR multiple unused formatting arguments

    // A specifier that cannot be translated keeps getting a note pointing at the whole `concat!`
    // invocation, as before.
    format_args!(concat!("%5.3d", ""), 1);
    //~^ ERROR argument never used
}
