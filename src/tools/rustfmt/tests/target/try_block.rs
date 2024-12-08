// rustfmt-edition: 2018

fn main() -> Result<(), !> {
    let _x: Option<_> = try { 4 };

    try {}
}

fn baz() -> Option<i32> {
    if (1 == 1) {
        return try { 5 };
    }

    // test
    let x: Option<()> = try {
        // try blocks are great
    };

    let y: Option<i32> = try { 6 }; // comment

    let x: Option<i32> = try {
        baz()?;
        baz()?;
        baz()?;
        7
    };

    return None;
}
