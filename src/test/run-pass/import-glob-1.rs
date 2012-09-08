use a1::b1::word_traveler;

mod a1 {
    //
    mod b1 {
        //
        use a2::b1::*;
        //         <-\
        export word_traveler; //           |
    }
    //           |
    mod b2 {
        //           |
        use a2::b2::*;
        // <-\  -\   |
        export word_traveler; //   |   |   |
    } //   |   |   |
}
//   |   |   |
//   |   |   |
mod a2 {
    //   |   |   |
    #[abi = "cdecl"]
    #[nolink]
    extern mod b1 {
        //   |   |   |
        use a1::b2::*;
        //   | <-/  -/
        export word_traveler; //   |
    }
    //   |
    mod b2 {
        //   |
        fn word_traveler() { //   |
            debug!("ahoy!"); //  -/
        } //
    } //
}
//


fn main() { word_traveler(); }
