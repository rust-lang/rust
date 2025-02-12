#![warn(clippy::wildcard_in_or_patterns)]

fn main() {
    match "foo" {
        "a" => {
            dbg!("matched a");
        },
        "bar" | _ => {
            //~^ wildcard_in_or_patterns

            dbg!("matched (bar or) wild");
        },
    };
    match "foo" {
        "a" => {
            dbg!("matched a");
        },
        "bar" | "bar2" | _ => {
            //~^ wildcard_in_or_patterns

            dbg!("matched (bar or bar2 or) wild");
        },
    };
    match "foo" {
        "a" => {
            dbg!("matched a");
        },
        _ | "bar" | _ => {
            //~^ wildcard_in_or_patterns

            dbg!("matched (bar or) wild");
        },
    };
    match "foo" {
        "a" => {
            dbg!("matched a");
        },
        _ | "bar" => {
            //~^ wildcard_in_or_patterns

            dbg!("matched (bar or) wild");
        },
    };

    // shouldn't lint
    #[non_exhaustive]
    pub enum NonExhaustiveEnum<'a> {
        Message(&'a str),
        Quit(&'a str),
        Other,
    }

    match NonExhaustiveEnum::Message("Pass") {
        NonExhaustiveEnum::Message(_) => dbg!("message"),
        NonExhaustiveEnum::Quit(_) => dbg!("quit"),
        NonExhaustiveEnum::Other | _ => dbg!("wildcard"),
    };

    // should lint
    enum ExhaustiveEnum {
        Quit,
        Write(String),
        ChangeColor(i32, i32, i32),
    }

    match ExhaustiveEnum::ChangeColor(0, 160, 255) {
        ExhaustiveEnum::Write(text) => {
            dbg!("Write");
        },
        ExhaustiveEnum::ChangeColor(r, g, b) => {
            dbg!("Change the color");
        },
        ExhaustiveEnum::Quit | _ => {
            //~^ wildcard_in_or_patterns
            dbg!("Quit or other");
        },
    };

    // shouldn't lint
    #[non_exhaustive]
    struct NonExhaustiveStruct {
        a: u32,
        b: u32,
        c: u64,
    }

    let b = NonExhaustiveStruct { a: 5, b: 42, c: 342 };

    match b {
        NonExhaustiveStruct { a: 5, b: 42, .. } => {},
        NonExhaustiveStruct { a: 0, b: 0, c: 128 } => {},
        NonExhaustiveStruct { a: 0, b: 0, c: 128, .. } | _ => {},
    }

    // should lint
    struct ExhaustiveStruct {
        x: i32,
        y: i32,
    }

    let p = ExhaustiveStruct { x: 0, y: 7 };
    match p {
        ExhaustiveStruct { x: 0, y: 0 } => {
            dbg!("On the x axis at {x}");
        },
        ExhaustiveStruct { x: 0, y: 1 } => {
            dbg!("On the y axis at {y}");
        },
        ExhaustiveStruct { x: 1, y: 1 } | _ => {
            //~^ wildcard_in_or_patterns
            dbg!("On neither axis: ({x}, {y})");
        },
    }
}
