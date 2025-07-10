#[derive(A)]
struct S {
    d: [u32; {
        #![cfg] {
            #![w,)
                   //~ ERROR this file contains an unclosed delimiter
