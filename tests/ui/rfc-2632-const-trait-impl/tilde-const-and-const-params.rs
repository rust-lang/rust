#![feature(const_trait_impl)]
#![feature(generic_arg_infer)]
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

struct Foo<const N: usize>;

impl<const N: usize> Foo<N> {
   fn add<A: ~const Add42>(self) -> Foo<{ A::add(N) }> {
      Foo
   }
}

#[const_trait]
trait Add42 {
    fn add(a: usize) -> usize;
}

impl const Add42 for () {
    fn add(a: usize) -> usize {
        a + 42
    }
}

fn bar<A: ~const Add42, const N: usize>(_: Foo<N>) -> Foo<{ A::add(N) }> {
    //~^ ERROR `~const` is not allowed here
    Foo
}

fn main() {
   let foo = Foo::<0>;
   let foo = bar::<(), _>(foo);
   let _foo = bar::<(), _>(foo);
}
