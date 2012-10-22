trait x {
    fn use_x<T>();
}
enum y = (); 
impl y:x { 
    fn use_x<T>() {
        struct foo { //~ ERROR quux
            i: ()
        }
        fn new_foo<T>(i: ()) -> foo {
            foo { i: i }
        }
    }   
}

