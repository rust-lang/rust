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
    //~^ ERROR dangling pointer

    // MethodCall
    "hello".to_string().as_ptr();
    //~^ ERROR dangling pointer

    // Tup
    // impossible

    // Binary
    (string() + "hello").as_ptr();
    //~^ ERROR dangling pointer

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
        //~^ ERROR dangling pointer
    }

    // Loop
    {
        (loop {
            break String::new();
        })
        .as_ptr();
        //~^ ERROR dangling pointer
    }

    // Match
    {
        match string() {
            s => s,
        }
        .as_ptr();
        //~^ ERROR dangling pointer
    }

    // Closure
    // impossible

    // Block
    { string() }.as_ptr();
    //~^ ERROR dangling pointer

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
    //~^ ERROR dangling pointer
}
