fn main() {}

const fn unsize(x: &[u8; 3]) -> &[u8] { x }
const fn closure() -> fn() { || {} }
//~^ ERROR function pointer
//~| ERROR function pointer cast
const fn closure2() {
    (|| {}) as fn();
//~^ ERROR function pointer
}
const fn reify(f: fn()) -> unsafe fn() { f }
//~^ ERROR function pointer
//~| ERROR function pointer
//~| ERROR function pointer cast
const fn reify2() { main as unsafe fn(); }
//~^ ERROR function pointer
//~| ERROR function pointer cast
