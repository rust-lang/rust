fn avg<T=T::Item>(_: T) {}
//~^ ERROR generic parameters with a default cannot use forward declared identifiers
//~| ERROR defaults for type parameters
//~| WARN previously accepted

fn main() {}
