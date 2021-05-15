fn main() {
    let x = if true {
        1
            // In if
    } else {
        0
            // In else
    };

    let x = if true {
        1
             /* In if */
    } else {
        0
             /* In else */
    };

    let z = if true {
        if true {
            1

                 // In if level 2
        } else {
            2
        }
    } else {
        3
    };

    let a = if true {
        1
  // In if
    } else {
        0
  // In else
    };

    let a = if true {
        1

    // In if
    } else {
        0
    // In else
    };

    let b = if true {
        1

    // In if
    } else {
        0
        // In else
    };

    let c = if true {
        1

        // In if
    } else {
        0
        // In else
    };
    for i in 0..2 {
        println!("Something");
        // In for
    }

    for i in 0..2 {
        println!("Something");
        /* In for */
    }

    extern "C" {
        fn first();

        // In foreign mod
    }

    extern "C" {
        fn first();

        /* In foreign mod */
    }
}
