fn foo(f: fn() -> !) {}

fn main() {
    // Type inference didn't use to be able to handle this:
    foo(|| die!());
    foo(|| 22); //~ ERROR mismatched types
}
