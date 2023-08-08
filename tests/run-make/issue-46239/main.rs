fn project<T>(x: &(T,)) -> &T { &x.0 }

fn dummy() {}

fn main() {
    let f = (dummy as fn(),);
    (*project(&f))();
}
