#[derive(A)]
struct S {
    d: [u32; {
        #![cfg] {
            #![w,) //~ ERROR mismatched closing delimiter
                   //~ ERROR this file contains an unclosed delimiter
