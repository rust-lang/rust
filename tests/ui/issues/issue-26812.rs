fn avg<T=T::Item>(_: T) {}
//~^ ERROR generic parameter defaults cannot reference parameters before they are declared
//~| ERROR defaults for type parameters
//~| WARN previously accepted

fn main() {}
