mod a {
    struct A;

    impl Default for A {
        pub fn default() -> A { //~ ERROR visibility qualifiers are not permitted here
            A
        }
    }
}


fn main() {
    a::A::default();
    //~^ ERROR struct `A` is private
 }
