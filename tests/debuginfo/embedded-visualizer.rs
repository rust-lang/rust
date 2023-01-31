// compile-flags:-g
// min-gdb-version: 8.1
// ignore-lldb
// ignore-windows-gnu // emit_debug_gdb_scripts is disabled on Windows

// === CDB TESTS ==================================================================================

// cdb-command: g

// The .nvlist command in cdb does not always have a deterministic output
// for the order that NatVis files are displayed.

// cdb-command: .nvlist
// cdb-check:    [...].exe (embedded NatVis "[...]embedded_visualizer-0.natvis")

// cdb-command: .nvlist
// cdb-check:    [...].exe (embedded NatVis "[...]embedded_visualizer-1.natvis")

// cdb-command: .nvlist
// cdb-check:    [...].exe (embedded NatVis "[...]embedded_visualizer-2.natvis")

// cdb-command: dx point_a
// cdb-check:point_a          : (0, 0) [Type: embedded_visualizer::point::Point]
// cdb-check:    [<Raw View>]     [Type: embedded_visualizer::point::Point]
// cdb-check:    [x]              : 0 [Type: int]
// cdb-check:    [y]              : 0 [Type: int]

// cdb-command: dx point_b
// cdb-check:point_b          : (5, 8) [Type: embedded_visualizer::point::Point]
// cdb-check:    [<Raw View>]     [Type: embedded_visualizer::point::Point]
// cdb-check:    [x]              : 5 [Type: int]
// cdb-check:    [y]              : 8 [Type: int]

// cdb-command: dx line
// cdb-check:line             : ((0, 0), (5, 8)) [Type: embedded_visualizer::Line]
// cdb-check:    [<Raw View>]     [Type: embedded_visualizer::Line]
// cdb-check:    [a]              : (0, 0) [Type: embedded_visualizer::point::Point]
// cdb-check:    [b]              : (5, 8) [Type: embedded_visualizer::point::Point]

// cdb-command: dx person
// cdb-check:person           : "Person A" is 10 years old. [Type: dependency_with_embedded_visualizers::Person]
// cdb-check:    [<Raw View>]     [Type: dependency_with_embedded_visualizers::Person]
// cdb-check:    [name]           : "Person A" [Type: alloc::string::String]
// cdb-check:    [age]            : 10 [Type: int]

// === GDB TESTS ===================================================================================

// gdb-command: run

// gdb-command: info auto-load python-scripts
// gdb-check:Yes     pretty-printer-embedded_visualizer-0
// gdb-check:Yes     pretty-printer-embedded_visualizer-1
// gdb-command: print point_a
// gdb-check:$1 = (0, 0)
// gdb-command: print point_b
// gdb-check:$2 = (5, 8)
// gdb-command: print line
// gdb-check:$3 = ((0, 0), (5, 8))
// gdb-command: print person
// gdb-check:$4 = "Person A" is 10 years old.

#![allow(unused_variables)]
#![feature(debugger_visualizer)]
#![debugger_visualizer(natvis_file = "embedded-visualizer.natvis")]
#![debugger_visualizer(gdb_script_file = "embedded-visualizer.py")]

// aux-build: dependency-with-embedded-visualizers.rs
extern crate dependency_with_embedded_visualizers;

use dependency_with_embedded_visualizers::Person;

#[debugger_visualizer(natvis_file = "embedded-visualizer-point.natvis")]
#[debugger_visualizer(gdb_script_file = "embedded-visualizer-point.py")]
mod point {
    pub struct Point {
        x: i32,
        y: i32,
    }

    impl Point {
        pub fn new(x: i32, y: i32) -> Point {
            Point { x: x, y: y }
        }
    }
}

use point::Point;

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

    let name = String::from("Person A");
    let person = Person::new(name, 10);

    zzz(); // #break
}

fn zzz() {
    ()
}
