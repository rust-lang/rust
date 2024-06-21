fn foo<T>() where <T>::Item: ToString, T: Iterator { }
//~^ ERROR generic parameters on `where` clauses are reserved for future use
//~| ERROR cannot find type `Item`

fn main() {}
