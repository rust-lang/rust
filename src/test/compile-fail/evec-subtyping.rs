fn wants_box(x: @[uint]) { }
fn wants_uniq(x: ~[uint]) { }
fn wants_three(x: [uint * 3]) { }

fn has_box(x: @[uint]) {
   wants_box(x);
   wants_uniq(x); //~ ERROR [] storage differs: expected ~ but found @
   wants_three(x); //~ ERROR [] storage differs: expected 3 but found @
}

fn has_uniq(x: ~[uint]) {
   wants_box(x); //~ ERROR [] storage differs: expected @ but found ~
   wants_uniq(x);
   wants_three(x); //~ ERROR [] storage differs: expected 3 but found ~
}

fn has_three(x: [uint * 3]) {
   wants_box(x); //~ ERROR [] storage differs: expected @ but found 3
   wants_uniq(x); //~ ERROR [] storage differs: expected ~ but found 3
   wants_three(x);
}

fn has_four(x: [uint * 4]) {
   wants_box(x); //~ ERROR [] storage differs: expected @ but found 4
   wants_uniq(x); //~ ERROR [] storage differs: expected ~ but found 4
   wants_three(x); //~ ERROR [] storage differs: expected 3 but found 4
}

fn main() {
}
