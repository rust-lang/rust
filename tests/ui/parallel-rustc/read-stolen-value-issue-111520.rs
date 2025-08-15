// Test for #111520, which causes an ice bug cause of reading stolen value
//
//@ compile-flags: -Z threads=16
//@ run-pass
//@ compare-output-by-lines

#[repr(transparent)]
struct Sched {
    i: i32,
}
impl Sched {
    extern "C" fn get(self) -> i32 {
        self.i
    }
}

fn main() {
    let s = Sched { i: 4 };
    let f = || -> i32 { s.get() };
    println!("f: {}", f());
}
