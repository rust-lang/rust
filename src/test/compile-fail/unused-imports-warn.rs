// error-pattern:unused import
// compile-flags:--warn-unused-imports
import cal = bar::c::cc;

mod foo {
    type point = {x: int, y: int};
    type square = {p: point, h: uint, w: uint};
}

mod bar {
    mod c {
        import foo::point;
        import foo::square;
        fn cc(p: point) -> str { ret 2 * (p.x + p.y); }
    }
}

fn main() {
    cal({x:3, y:9});
}
