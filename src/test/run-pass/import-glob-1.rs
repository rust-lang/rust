import a1::b1::word_traveler;

mod a1 {
     //
    mod b1 {
         //
        import a2::b1::*;
         //         <-\
        export word_traveler; //           |
    }
     //           |
    mod b2 {
         //           |
        import a2::b2::*;
         // <-\  -\   |
        export word_traveler; //   |   |   |
    } //   |   |   |
}
 //   |   |   |
 //   |   |   |
 mod a2 {
      //   |   |   |
     native "cdecl" mod b1 = "" {
          //   |   |   |
         import a1::b2::*;
          //   | <-/  -/
         export word_traveler; //   |
     }
      //   |
     mod b2 {
          //   |
         fn word_traveler() { //   |
             log "ahoy!"; //  -/
         } //
     } //
 }
 //


fn main() { word_traveler(); }