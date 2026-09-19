fn block() {
    let inner = {
        #![doc("Inner attributes not allowed here")]
        //! Nor are ModuleDoc comments
    };
    if true {
        #![doc("Nor here")]
        #![doc("We error on each attr")]
        //! Nor are ModuleDoc comments
    }
    while true {
        #![doc("Nor here")]
        //! Nor are ModuleDoc comments
    }
     loop {
        #![doc("This is fine, `loop` bodies accept inner attributes")]
        //! So are ModuleDoc comments
    }
    for _ in 0..1 {
        #![doc("This is fine, `for` bodies accept inner attributes")]
        //! So are ModuleDoc comments
    }
    let t = (
        {
            #![doc("This is fine, tuple elements accept inner attributes")]
            //! So are ModuleDoc comments
        },
    );
    let a = [
        {
            #![doc("This is fine, array elements accept inner attributes")]
            //! So are ModuleDoc comments
        },
    ];
    g({
        #![doc("This is fine, call arguments accept inner attributes")]
        //! So are ModuleDoc comments
    });
    s.m({
        #![doc("This is fine, method call arguments accept attributes")]
        //! So are ModuleDoc comments
    });
}
