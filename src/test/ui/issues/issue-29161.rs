mod a {
    struct A;

    impl Default for A {
        pub fn default() -> A { //~ ERROR unnecessary visibility qualifier
            A
        }
    }
}


fn main() {
    a::A::default();
    //~^ ERROR struct `A` is private
 }
