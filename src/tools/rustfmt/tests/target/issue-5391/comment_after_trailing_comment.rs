//@ rustfmt-style_edition: 2024

fn custom_comments() {
    // Custom comment is the last element of the block
    {
        let x = 0; // X
        //.Y
    }

    // Custom comment is not the last element of the block
    {
        let x = 0; // X
        //.Y
        println!("hello world!");
    }

    // Variations on custom comment tests with and without extra text
    {
        let x = 0; // X
        //~Y
    }
    {
        let x = 0; // X
        //~Y
        println!("hello world!");
    }
    {
        let x = 0; // X
        //.Y Some more text
    }
    {
        let x = 0; // X
        //.Y Some more text
        println!("hello world!");
    }
    {
        let x = 0; // X
        //~Y Some more text
    }
    {
        let x = 0; // X
        //~Y Some more text
        println!("hello world!");
    }
}

fn double_slash_comments() {
    // DoubleSlash comment is the last element of the block
    {
        let x = 0; // X
        // .Y
    }

    // DoubleSlash comment is not the the last element of the block
    {
        let x = 0; // X
        // .Y
        println!("hello world!");
    }
    {
        let x = 0; // X
        // ~Y
    }
    {
        let x = 0; // X
        // ~Y
        println!("hello world!");
    }
    {
        let x = 0; // X
        //Y
    }
    {
        let x = 0; // X
        //Y
        println!("hello world!");
    }
    {
        let x = 0; // X
        // Y
    }
    {
        let x = 0; // X
        // Y
        println!("hello world!");
    }
}

// Compiler UI Test Error Annotations that use custom comments
// https://rustc-dev-guide.rust-lang.org/tests/ui.html#error-annotation-examples
mod compiler_ui_test_error_annotations {
    /// Use the //~ ERROR idiom:
    fn positioned_on_error_line() {
        let x = (1, 2, 3);
        match x {
            (_a, _x @ ..) => {} //~ ERROR `_x @` is not allowed in a tuple
            _ => {}
        }
    }

    /// Use the //~^ idiom with number of carets in the string to indicate the number of lines above
    fn positioned_below_error_line() {
        let x = (1, 2, 3);
        match x {
            (_a, _x @ ..) => {} // <- the error is on this line
            _ => {}
        }
        //~^^^ ERROR `_x @` is not allowed in a tuple
    }

    struct Binder(i32, i32, i32);

    /// Use the //~| idiom to define the same error line as the error annotation line above
    fn use_same_error_line_as_defined_on_error_annotation_line_above() {
        let x = Binder(1, 2, 3);
        match x {
            Binder(_a, _x @ ..) => {} // <- the error is on this line
            _ => {}
        }
        //~^^^ ERROR `_x @` is not allowed in a tuple struct
        //~| ERROR this pattern has 1 field, but the corresponding tuple struct has 3 fields [E0023]
    }
}
