//@ run-rustfix
macro_rules! m {
    (begin $($ard:ident),* end) => {
        [$($arg),*] //~ ERROR attempted to repeat an expression containing no syntax variables matched as repeating at this depth
        //~^ HELP there's a macro metavariable with a similar name
    };
}

fn main() {
    let x = 1;
    let _ = m![begin x end];
}
