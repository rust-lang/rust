// FIXME(calebcartwright) - Hopefully one day we can
// elide these redundant semis like we do in other contexts.
fn redundant_item_semis() {
    impl Foo {
            fn get(&self) -> usize {
                5
            }
        };
        
            impl Bar {
            fn get(&self) -> usize {
                5
            }
        } /*asdfsf*/;
        
        
    impl Baz {
        fn get(&self) -> usize {
            5
        }
    } /*asdfsf*/
    
    // why would someone do this
    ;
    
    
        impl Qux {
        fn get(&self) -> usize {
            5
        }
    } 
    
    // why
    ;
    
            impl Lorem {
        fn get(&self) -> usize {
            5
        }
    } 
    // oh why
    ;
}