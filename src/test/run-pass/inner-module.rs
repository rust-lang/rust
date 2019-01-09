mod inner {
    pub mod inner2 {
        pub fn hello() { println!("hello, modular world"); }
    }
    pub fn hello() { inner2::hello(); }
}

pub fn main() { inner::hello(); inner::inner2::hello(); }
