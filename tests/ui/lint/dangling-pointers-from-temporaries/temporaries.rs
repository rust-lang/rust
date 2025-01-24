#![allow(unused)]
#![deny(dangling_pointers_from_temporaries)]

fn string() -> String {
    "hello".into()
}

struct Wrapper(String);

fn main() {
    // ConstBlock
    const { String::new() }.as_ptr();

    // Array
    {
        [string()].as_ptr(); // False negative
        [true].as_ptr();
    }

    // Call
    string().as_ptr();
    //~^ ERROR a dangling pointer will be produced because the temporary `String` will be dropped

    // MethodCall
    "hello".to_string().as_ptr();
    //~^ ERROR a dangling pointer will be produced because the temporary `String` will be dropped

    // Tup
    // impossible

    // Binary
    (string() + "hello").as_ptr();
    //~^ ERROR a dangling pointer will be produced because the temporary `String` will be dropped

    // Path
    {
        let x = string();
        x.as_ptr();
    }

    // Unary
    {
        let x = string();
        let x: &String = &x;
        (*x).as_ptr();
        (&[0u8]).as_ptr();
        (&string()).as_ptr(); // False negative
        (*&string()).as_ptr(); // False negative
    }

    // Lit
    "hello".as_ptr();

    // Cast
    // impossible

    // Type
    // impossible

    // DropTemps
    // impossible

    // Let
    // impossible

    // If
    {
        (if true { String::new() } else { "hello".into() }).as_ptr();
        //~^ ERROR a dangling pointer will be produced because the temporary `String` will be dropped
    }

    // Loop
    {
        (loop {
            break String::new();
        })
        .as_ptr();
        //~^ ERROR a dangling pointer will be produced because the temporary `String` will be dropped
    }

    // Match
    {
        match string() {
            s => s,
        }
        .as_ptr();
        //~^ ERROR a dangling pointer will be produced because the temporary `String` will be dropped
    }

    // Closure
    // impossible

    // Block
    { string() }.as_ptr();
    //~^ ERROR a dangling pointer will be produced because the temporary `String` will be dropped

    // Assign, AssignOp
    // impossible

    // Field
    {
        Wrapper(string()).0.as_ptr(); // False negative
        let x = Wrapper(string());
        x.0.as_ptr();
    }

    // Index
    {
        vec![string()][0].as_ptr(); // False negative
        let x = vec![string()];
        x[0].as_ptr();
    }

    // AddrOf, InlineAsm, OffsetOf
    // impossible

    // Break, Continue, Ret
    // are !

    // Become, Yield
    // unstable, are !

    // Repeat
    [0u8; 100].as_ptr();
    [const { String::new() }; 100].as_ptr();

    // Struct
    // Cannot test this without access to private fields of the linted types.

    // Err
    // impossible

    // Macro
    vec![0u8].as_ptr();
    //~^ ERROR a dangling pointer will be produced because the temporary `Vec<u8>` will be dropped
}
