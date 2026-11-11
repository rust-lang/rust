trait A { fn foo(&self); }
trait B { fn foo(&self); }

fn foo<T:A + B>(t: T) {
    t.foo(); //~ ERROR E0034
}

fn main() {}
