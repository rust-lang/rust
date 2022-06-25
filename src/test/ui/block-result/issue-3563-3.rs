// run-pass
#![allow(unused_imports)]
#![allow(non_snake_case)]

// ASCII art shape renderer.  Demonstrates traits, impls, operator overloading,
// non-copyable struct, unit testing.  To run execute: rustc --test shapes.rs &&
// ./shapes

// Rust's std library is tightly bound to the language itself so it is
// automatically linked in.  However the extra library is designed to be
// optional (for code that must run on constrained environments like embedded
// devices or special environments like kernel code) so it must be explicitly
// linked in.

// Extern mod controls linkage. Use controls the visibility of names to modules
// that are already linked in. Using WriterUtil allows us to use the write_line
// method.

use std::fmt;
use std::iter::repeat;
use std::slice;

// Represents a position on a canvas.
#[derive(Copy, Clone)]
struct Point {
    x: isize,
    y: isize,
}

// Represents an offset on a canvas. (This has the same structure as a Point.
// but different semantics).
#[derive(Copy, Clone)]
struct Size {
    width: isize,
    height: isize,
}

#[derive(Copy, Clone)]
struct Rect {
    top_left: Point,
    size: Size,
}

// Contains the information needed to do shape rendering via ASCII art.
struct AsciiArt {
    width: usize,
    height: usize,
    fill: char,
    lines: Vec<Vec<char> > ,

    // This struct can be quite large so we'll disable copying: developers need
    // to either pass these structs around via references or move them.
}

impl Drop for AsciiArt {
    fn drop(&mut self) {}
}

// It's common to define a constructor sort of function to create struct instances.
// If there is a canonical constructor it is typically named the same as the type.
// Other constructor sort of functions are typically named from_foo, from_bar, etc.
fn AsciiArt(width: usize, height: usize, fill: char) -> AsciiArt {
    // Build a vector of vectors containing blank characters for each position in
    // our canvas.
    let lines = vec![vec!['.'; width]; height];

    // Rust code often returns values by omitting the trailing semi-colon
    // instead of using an explicit return statement.
    AsciiArt {width: width, height: height, fill: fill, lines: lines}
}

// Methods particular to the AsciiArt struct.
impl AsciiArt {
    fn add_pt(&mut self, x: isize, y: isize) {
        if x >= 0 && x < self.width as isize {
            if y >= 0 && y < self.height as isize {
                // Note that numeric types don't implicitly convert to each other.
                let v = y as usize;
                let h = x as usize;

                // Vector subscripting will normally copy the element, but &v[i]
                // will return a reference which is what we need because the
                // element is:
                // 1) potentially large
                // 2) needs to be modified
                let row = &mut self.lines[v];
                row[h] = self.fill;
            }
        }
    }
}

// Allows AsciiArt to be converted to a string using the libcore ToString trait.
// Note that the %s fmt! specifier will not call this automatically.
impl fmt::Display for AsciiArt {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Convert each line into a string.
        let lines = self.lines.iter()
                              .map(|line| line.iter().cloned().collect())
                              .collect::<Vec<String>>();

        // Concatenate the lines together using a new-line.
        write!(f, "{}", lines.join("\n"))
    }
}

// This is similar to an interface in other languages: it defines a protocol which
// developers can implement for arbitrary concrete types.
trait Canvas {
    fn add_point(&mut self, shape: Point);
    fn add_rect(&mut self, shape: Rect);

    // Unlike interfaces traits support default implementations.
    // Got an ICE as soon as I added this method.
    fn add_points(&mut self, shapes: &[Point]) {
        for pt in shapes {self.add_point(*pt)};
    }
}

// Here we provide an implementation of the Canvas methods for AsciiArt.
// Other implementations could also be provided (e.g., for PDF or Apple's Quartz)
// and code can use them polymorphically via the Canvas trait.
impl Canvas for AsciiArt {
    fn add_point(&mut self, shape: Point) {
        self.add_pt(shape.x, shape.y);
    }

    fn add_rect(&mut self, shape: Rect) {
        // Add the top and bottom lines.
        for x in shape.top_left.x..shape.top_left.x + shape.size.width {
            self.add_pt(x, shape.top_left.y);
            self.add_pt(x, shape.top_left.y + shape.size.height - 1);
        }

        // Add the left and right lines.
        for y in shape.top_left.y..shape.top_left.y + shape.size.height {
            self.add_pt(shape.top_left.x, y);
            self.add_pt(shape.top_left.x + shape.size.width - 1, y);
        }
    }
}

// Rust's unit testing framework is currently a bit under-developed so we'll use
// this little helper.
pub fn check_strs(actual: &str, expected: &str) -> bool {
    if actual != expected {
        println!("Found:\n{}\nbut expected\n{}", actual, expected);
        return false;
    }
    return true;
}


fn test_ascii_art_ctor() {
    let art = AsciiArt(3, 3, '*');
    assert!(check_strs(&art.to_string(), "...\n...\n..."));
}


fn test_add_pt() {
    let mut art = AsciiArt(3, 3, '*');
    art.add_pt(0, 0);
    art.add_pt(0, -10);
    art.add_pt(1, 2);
    assert!(check_strs(&art.to_string(), "*..\n...\n.*."));
}


fn test_shapes() {
    let mut art = AsciiArt(4, 4, '*');
    art.add_rect(Rect {top_left: Point {x: 0, y: 0}, size: Size {width: 4, height: 4}});
    art.add_point(Point {x: 2, y: 2});
    assert!(check_strs(&art.to_string(), "****\n*..*\n*.**\n****"));
}

pub fn main() {
    test_ascii_art_ctor();
    test_add_pt();
    test_shapes();
}
