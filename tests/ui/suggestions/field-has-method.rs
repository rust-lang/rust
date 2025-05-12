struct Kind;

struct Ty {
    kind: Kind,
}

impl Ty {
    fn kind(&self) -> Kind {
        todo!()
    }
}

struct InferOk<T> {
    value: T,
    predicates: Vec<()>,
}

fn foo(i: InferOk<Ty>) {
    let k = i.kind();
    //~^ ERROR no method named `kind` found for struct `InferOk` in the current scope
}

fn main() {}
