// error-pattern:unused import
// compile-flags:-W unused-imports
use cal = bar::c::cc;

mod foo {
    #[legacy_exports];
    type point = {x: int, y: int};
    type square = {p: point, h: uint, w: uint};
}

mod bar {
    #[legacy_exports];
    mod c {
        #[legacy_exports];
        use foo::point;
        use foo::square;
        fn cc(p: point) -> str { return 2 * (p.x + p.y); }
    }
}

fn main() {
    cal({x:3, y:9});
}
