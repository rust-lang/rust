struct S {
    x: u8,
}

mod m {
    trait GetX {
        fn x(&self) -> u8;
    }

    impl GetX for super::S {
        fn x(&self) -> u8 {
            self.x
        }
    }
}

pub fn show_x(st: S) {
    println!("{}", st.x())
    //~^ ERROR no method named `x` found for struct `S` in the current scope [E0599]

}

fn main() {}
