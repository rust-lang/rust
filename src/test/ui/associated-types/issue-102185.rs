// compile-flags: -Zdeduplicate-diagnostics=yes

fn f<T>(_: impl A<X = T>) {}
                //~^ ERROR the trait bound `T: B` is not satisfied [E0277]
trait A: C<Z = <<Self as A>::X as B>::Y> {
    type X: B;
}

trait B {
    type Y;
}

trait C {
    type Z;
}

fn main() {}
