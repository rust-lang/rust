// Regression test for #156101.
//
// When the format string comes from `concat!`, the parser's inner offsets are relative to the
// expanded string, not to the source, so they must not be turned into spans in the source.
// Before the fix, suggestions built from those offsets went wrong in two ways:
//
// - If the span landed inside a multibyte character, rendering the suggestion caused an ICE.
//   The `ReorderFormatParameter` and `AddMissingColon` cases below are padded so that the span
//   falls inside `é` (2 bytes) or `𐏿` (4 bytes).
// - If the span landed on unrelated source text, `UsePositional` suggested a bogus argument
//   copied from that text.
//
// After the fix, no suggestion is emitted for these format strings.

fn main() {
    // The original reproducer from the issue.
    format_args!(concat!("𐏿", "{f:?#}"));
    //~^ ERROR invalid format string

    // `Suggestion::ReorderFormatParameter`
    format_args!(concat!("é", "aaa{:?#}"), 1);
    //~^ ERROR invalid format string
    format_args!(concat!("𐏿", "a{:?#}"), 1);
    //~^ ERROR invalid format string

    // `Suggestion::AddMissingColon`
    format_args!(concat!("é", "aaaa{x?}"));
    //~^ ERROR invalid format string
    format_args!(concat!("𐏿", "aa{x?}"));
    //~^ ERROR invalid format string

    // `Suggestion::UsePositional`: these did not ICE, but before the fix they suggested a bogus
    // argument taken from the `concat!` invocation instead of from the format string.
    format_args!(concat!("{}{a.b}", ""));
    //~^ ERROR invalid format string
    format_args!(concat!("{}{a.0}", ""));
    //~^ ERROR invalid format string
}
