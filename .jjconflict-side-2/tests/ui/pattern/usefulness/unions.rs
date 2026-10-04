fn main() {
    #[derive(Copy, Clone)]
    union U8AsBool {
        n: u8,
        b: bool,
    }

    let x = U8AsBool { n: 1 };
    unsafe {
        match x {
            // exhaustive
            U8AsBool { n: 2 } => {}
            U8AsBool { b: true } => {}
            U8AsBool { b: false } => {}
        }
        match x {
            // exhaustive
            U8AsBool { b: true } => {}
            U8AsBool { n: 0 } => {}
            U8AsBool { n: 1.. } => {}
        }
        match x {
            //~^ ERROR non-exhaustive patterns: `U8AsBool { n: 0_u8 }` and `U8AsBool { b: false }` not covered
            U8AsBool { b: true } => {}
            U8AsBool { n: 1.. } => {}
        }
        // Our approach can report duplicate witnesses sometimes.
        match (x, true) {
            //~^ ERROR non-exhaustive patterns: `(U8AsBool { n: 0_u8 }, false)`, `(U8AsBool { b: true }, false)`, `(U8AsBool { n: 0_u8 }, false)` and 1 more not covered
            (U8AsBool { b: true }, true) => {}
            (U8AsBool { b: false }, true) => {}
            (U8AsBool { n: 1.. }, true) => {}
        }
    }
}
