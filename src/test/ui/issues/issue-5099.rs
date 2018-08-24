trait B < A > { fn a() -> A { this.a } } //~ ERROR cannot find value `this` in this scope

fn main() {}
