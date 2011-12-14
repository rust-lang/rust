// Tests that the tail expr in null() has its type
// unified with the type *T, and so the type variable
// in that type gets resolved.
use std;
import unsafe;

fn null<T>() -> *T unsafe { unsafe::reinterpret_cast(0) }

fn main() { null::<int>(); }
