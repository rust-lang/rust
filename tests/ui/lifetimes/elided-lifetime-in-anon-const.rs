// Verify that elided lifetimes inside anonymous constants are not forced to be `'static`.
//@ check-pass

fn foo() -> [(); {
       let a = 10_usize;
       let b: &'_ usize = &a;
       *b
   }] {
    [(); 10]
}

fn bar() -> [(); 10] {
    [(); {
        let a = 10_usize;
        let b: &'_ usize = &a;
        *b
    }]
}

fn main() {}
