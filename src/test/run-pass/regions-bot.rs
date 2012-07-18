// A very limited test of the "bottom" region

fn produce_static<T>() -> &static/T { fail; }

fn foo<T>(x: &T) -> &uint { produce_static() }

fn main() {
}