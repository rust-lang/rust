fn foo() -> impl ?Sized {
    //~^ ERROR the size for values of type `impl ?Sized` cannot be known at compilation time
    ()
}

fn main() {}
