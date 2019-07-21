fn main() {
    let _ = Option:Some(vec![0, 1]); //~ ERROR expected type, found
}

// This case isn't currently being handled gracefully due to the macro invocation.
