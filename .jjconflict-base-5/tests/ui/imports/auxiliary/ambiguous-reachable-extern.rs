mod m1 {
    pub fn generic<T>() {
        let x = 10;
        let y = 11;
        println!("hello {x} world {:?}", y);
    }
}

mod m2 {
    pub fn generic() {}
}

pub use m1::*;
pub use m2::*;
