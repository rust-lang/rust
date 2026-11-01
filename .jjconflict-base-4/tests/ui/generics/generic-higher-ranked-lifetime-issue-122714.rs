#![allow(dead_code)]

trait Trait1<T>
  where T: for<'a> Trait1<T> + 'b { } //~ ERROR use of undeclared lifetime name `'b`

trait Trait2<T>
where
    T: B<'b> + for<'a> A<'a>, //~ ERROR use of undeclared lifetime name `'b`
{
}

trait Trait3<T>
where
    T: B<'b> + for<'a> A<'a> + 'c {}
    //~^ ERROR use of undeclared lifetime name `'b`
    //~| ERROR use of undeclared lifetime name `'c`

trait Trait4<T>
where
    T: for<'a> A<'a> + 'x + for<'b> B<'b>, //~ ERROR use of undeclared lifetime name `'x`
{
}

trait A<'a> {}
trait B<'a> {}


fn main() {}
