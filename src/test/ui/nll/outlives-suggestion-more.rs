// Test the more elaborate outlives suggestions.

// Should suggest: 'a: 'c, 'b: 'd
fn foo1<'a, 'b, 'c, 'd>(x: &'a usize, y: &'b usize) -> (&'c usize, &'d usize) {
    (x, y) //~ERROR lifetime may not live long enough
           //~^ERROR lifetime may not live long enough
}

// Should suggest: 'a: 'c and use 'static instead of 'b
fn foo2<'a, 'b, 'c>(x: &'a usize, y: &'b usize) -> (&'c usize, &'static usize) {
    (x, y) //~ERROR lifetime may not live long enough
           //~^ERROR lifetime may not live long enough
}

// Should suggest: 'a and 'b are the same and use 'static instead of 'c
fn foo3<'a, 'b, 'c, 'd, 'e>(
    x: &'a usize,
    y: &'b usize,
    z: &'c usize,
) -> (&'b usize, &'a usize, &'static usize) {
    (x, y, z) //~ERROR lifetime may not live long enough
              //~^ERROR lifetime may not live long enough
              //~^^ERROR lifetime may not live long enough
}

fn main() {}
