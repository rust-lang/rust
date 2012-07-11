type point = { x: int, y: int };

trait operators {
    pure fn +(z: int) -> int;
    fn *(z: int) -> int;
    fn [](z: int) -> int;
    fn unary-() -> int;
}

impl foo of operators for point {
    // expr_binary
    pure fn +(z: int) -> int { self.x + self.y + z }
    fn *(z: int) -> int { self.x * self.y * z }

    // expr_index
    fn [](z: int) -> int { self.x * self.y * z }

    // expr_unary
    fn unary-() -> int { -(self.x * self.y) }
}

pure fn a(p: point) -> int { p + 3 }

pure fn b(p: point) -> int { p * 3 }
//~^ ERROR access to impure function prohibited in pure context

pure fn c(p: point) -> int { p[3] }
//~^ ERROR access to impure function prohibited in pure context

pure fn d(p: point) -> int { -p }
//~^ ERROR access to impure function prohibited in pure context

fn main() {}
