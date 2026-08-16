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
}
