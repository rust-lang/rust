// run-pass
// See Issues #20055 and #21695.

// We are checking here that the temporaries `Box<[i8, k]>`, for `k`
// in 1, 2, 3, 4, that are induced by the match expression are
// properly handled, in that only *one* will be initialized by
// whichever arm is run, and subsequently dropped at the end of the
// statement surrounding the `match`.

trait Boo {
    fn dummy(&self) { }
}

impl Boo for [i8; 1] { }
impl Boo for [i8; 2] { }
impl Boo for [i8; 3] { }
impl Boo for [i8; 4] { }

pub fn foo(box_1: fn () -> Box<[i8; 1]>,
           box_2: fn () -> Box<[i8; 2]>,
           box_3: fn () -> Box<[i8; 3]>,
           box_4: fn () -> Box<[i8; 4]>,
            ) {
    println!("Hello World 1");
    let _: Box<dyn Boo> = match 3 {
        1 => box_1(),
        2 => box_2(),
        3 => box_3(),
        _ => box_4(),
    };
    println!("Hello World 2");
}

pub fn main() {
    fn box_1() -> Box<[i8; 1]> { Box::new( [1; 1] ) }
    fn box_2() -> Box<[i8; 2]> { Box::new( [1; 2] ) }
    fn box_3() -> Box<[i8; 3]> { Box::new( [1; 3] ) }
    fn box_4() -> Box<[i8; 4]> { Box::new( [1; 4] ) }

    foo(box_1, box_2, box_3, box_4);
}
