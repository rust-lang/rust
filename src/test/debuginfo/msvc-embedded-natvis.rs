// only-cdb
// compile-flags:-g

// === CDB TESTS ==================================================================================

// cdb-command: g

// cdb-command: .nvlist
// cdb-check:    [...].exe (embedded NatVis "[...]msvc_embedded_natvis-0.natvis")

// cdb-command: dx point_a
// cdb-check:point_a          : (0, 0) [Type: msvc_embedded_natvis::Point]
// cdb-check:    [<Raw View>]     [Type: msvc_embedded_natvis::Point]
// cdb-check:    [x]              : 0 [Type: int]
// cdb-check:    [y]              : 0 [Type: int]

// cdb-command: dx point_b
// cdb-check:point_b          : (5, 8) [Type: msvc_embedded_natvis::Point]
// cdb-check:    [<Raw View>]     [Type: msvc_embedded_natvis::Point]
// cdb-check:    [x]              : 5 [Type: int]
// cdb-check:    [y]              : 8 [Type: int]

// cdb-command: dx line
// cdb-check:line             : ((0, 0), (5, 8)) [Type: msvc_embedded_natvis::Line]
// cdb-check:    [<Raw View>]     [Type: msvc_embedded_natvis::Line]
// cdb-check:    [a]              : (0, 0) [Type: msvc_embedded_natvis::Point]
// cdb-check:    [b]              : (5, 8) [Type: msvc_embedded_natvis::Point]

#![feature(debugger_visualizer)]
#![debugger_visualizer(natvis_file = "msvc-embedded-natvis.natvis")]

pub struct Point {
    x: i32,
    y: i32,
}

impl Point {
    pub fn new(x: i32, y: i32) -> Point {
        Point { x: x, y: y }
    }
}

pub struct Line {
    a: Point,
    b: Point,
}

impl Line {
    pub fn new(a: Point, b: Point) -> Line {
        Line { a: a, b: b }
    }
}

fn main() {
    let point_a = Point::new(0, 0);
    let point_b = Point::new(5, 8);
    let line = Line::new(point_a, point_b);

    zzz(); // #break
}

fn zzz() {
    ()
}
