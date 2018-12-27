// run-pass
trait Equal {
    fn equal(&self, other: &Self) -> bool;
    fn equals<T,U>(&self, this: &T, that: &T, x: &U, y: &U) -> bool
            where T: Eq, U: Eq;
}

impl<T> Equal for T where T: Eq {
    fn equal(&self, other: &T) -> bool {
        self == other
    }
    fn equals<U,X>(&self, this: &U, other: &U, x: &X, y: &X) -> bool
            where U: Eq, X: Eq {
        this == other && x == y
    }
}

fn equal<T>(x: &T, y: &T) -> bool where T: Eq {
    x == y
}

fn main() {
    println!("{}", equal(&1, &2));
    println!("{}", equal(&1, &1));
    println!("{}", "hello".equal(&"hello"));
    println!("{}", "hello".equals::<isize,&str>(&1, &1, &"foo", &"bar"));
}
