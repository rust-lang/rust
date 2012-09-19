// Tests that the tail expr in null() has its type
// unified with the type *T, and so the type variable
// in that type gets resolved.
extern mod std;

fn null<T>() -> *T unsafe { cast::reinterpret_cast(&0) }

fn main() { null::<int>(); }
