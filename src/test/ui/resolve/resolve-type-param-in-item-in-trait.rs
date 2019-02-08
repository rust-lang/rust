// Issue #14603: Check for references to type parameters from the
// outer scope (in this case, the trait) used on items in an inner
// scope (in this case, the enum).

trait TraitA<A> {
    fn outer(&self) {
        enum Foo<B> {
            Variance(A)
                //~^ ERROR can't use generic parameters from outer function
        }
    }
}

trait TraitB<A> {
    fn outer(&self) {
        struct Foo<B>(A);
                //~^ ERROR can't use generic parameters from outer function
    }
}

trait TraitC<A> {
    fn outer(&self) {
        struct Foo<B> { a: A }
                //~^ ERROR can't use generic parameters from outer function
    }
}

trait TraitD<A> {
    fn outer(&self) {
        fn foo<B>(a: A) { }
                //~^ ERROR can't use generic parameters from outer function
    }
}

fn main() { }
