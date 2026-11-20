// Demonstrate that we still consider keyword `box` to begin expressions (`can_begin_expr`) even
// though we officially no longer support box expressions (#108471).
// It means that we take the first rule and fail immediately afterward.

// FIXME: Remove `box` from the list of tokens that can begin expressions which would make us take
//        the second rule instead and consequently accept this program (needs lang FCP).
//
//        Alternatively we could unreserve keyword `box` (needs lang FCP) which would make us
//        continue to take the first rule but also start accepting this program.

macro_rules! mk {
    ($e:expr) => {};
    (box $e:expr) => {};
}

mk!(box 0); //~ ERROR expected expression, found reserved keyword `box`

fn main() {}
