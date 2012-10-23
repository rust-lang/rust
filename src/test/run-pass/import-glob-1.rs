use a1::b1::word_traveler;

mod a1 {
    #[legacy_exports];
    //
    mod b1 {
        #[legacy_exports];
        //
        use a2::b1::*;
        //         = move\
        export word_traveler; //           |
    }
    //           |
    mod b2 {
        #[legacy_exports];
        //           |
        use a2::b2::*;
        // = move\  -\   |
        export word_traveler; //   |   |   |
    } //   |   |   |
}
//   |   |   |
//   |   |   |
mod a2 {
    #[legacy_exports];
    //   |   |   |
    #[abi = "cdecl"]
    #[nolink]
    extern mod b1 {
        #[legacy_exports];
        //   |   |   |
        use a1::b2::*;
        //   | = move/  -/
        export word_traveler; //   |
    }
    //   |
    mod b2 {
        #[legacy_exports];
        //   |
        fn word_traveler() { //   |
            debug!("ahoy!"); //  -/
        } //
    } //
}
//


fn main() { word_traveler(); }
