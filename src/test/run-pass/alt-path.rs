

mod m1 {
    tag foo { foo1; foo2; }
}

fn bar(m1::foo x) { alt (x) { case (m1::foo1) { } } }

fn main(vec[str] args) { }