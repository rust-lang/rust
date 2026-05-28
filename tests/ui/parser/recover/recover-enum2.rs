fn main() {
    enum Test {
        Var1,
        Var2(String),
        Var3 {
            abc: {}, //~ ERROR: expected type, found `{`
        },
    }

    // recover...
    let () = 1; //~ ERROR mismatched types
    enum Test2 {
        Fine,
    }

    enum Test3 {
        StillFine {
            def: i32,
        },
    }

    {
        // fail again
        enum Test4 {
            Nope(i32 {}) //~ ERROR: found `{`
        }
        let () = 1; //~ ERROR mismatched types
    }
}
