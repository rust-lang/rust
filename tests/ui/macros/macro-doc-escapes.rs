//@ run-pass
// When expanding a macro, documentation attributes (including documentation comments) must be
// passed "as is" without being parsed. Otherwise, some text will be incorrectly interpreted as
// escape sequences, leading to an ICE.
//
// Related issues: #25929, #25943

macro_rules! homura {
    (#[$x:meta]) => ()
}

homura! {
    /// \madoka \x41
}

fn main() { }
