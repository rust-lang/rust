mod x {
    pub struct A;
    pub struct B;
}

// `.` is similar to `,` so list parsing should continue to closing `}`
use x::{A. B}; //~ ERROR expected one of `,`, `::`, `as`, or `}`, found `.`

fn main() {}
