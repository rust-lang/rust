use std;

struct Deserializer : std::serialization::deserializer{ //~ ERROR obsolete syntax: class traits
    x: ()
}

type foo = {a: (),};

fn deserialize_foo<__D: std::serialization::deserializer>(&&__d: __D) {
}

fn main() { let des = Deserializer(); let foo = deserialize_foo(des); }
