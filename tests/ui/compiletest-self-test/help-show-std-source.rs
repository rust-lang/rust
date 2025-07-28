enum Foo<T> {
    Bar {
        v23: T,
        y: isize
    }
}

fn f(x: &Foo) { //~ ERROR missing generics for enum `Foo` [E0107]
    match *x {
        Foo::Bar { y: y, v23: x } => {
            assert_eq!(x, 1);
            assert_eq!(y, 2); //~ ERROR can't compare `&isize` with `{integer}` [E0277]
        }
    }
}

pub fn main() {
    let x = Foo::Bar { x: 1, y: 2 }; //~ ERROR variant `Foo<_>::Bar` has no field named `x` [E0559]
    f(&x);
}
