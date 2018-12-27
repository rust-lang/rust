fn main() {}

const fn unsize(x: &[u8; 3]) -> &[u8] { x }
//~^ ERROR unsizing casts are not allowed in const fn
const fn closure() -> fn() { || {} }
//~^ ERROR function pointers in const fn are unstable
const fn closure2() {
    (|| {}) as fn();
//~^ ERROR function pointers in const fn are unstable
}
const fn reify(f: fn()) -> unsafe fn() { f }
//~^ ERROR function pointers in const fn are unstable
const fn reify2() { main as unsafe fn(); }
//~^ ERROR function pointers in const fn are unstable
