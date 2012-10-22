fn wants_box(x: @str) { }
fn wants_uniq(x: ~str) { }
fn wants_slice(x: &str) { }

fn has_box(x: @str) {
   wants_box(x);
   wants_uniq(x); //~ ERROR str storage differs: expected ~ but found @
   wants_slice(x);
}

fn has_uniq(x: ~str) {
   wants_box(x); //~ ERROR str storage differs: expected @ but found ~
   wants_uniq(x);
   wants_slice(x);
}

fn has_slice(x: &str) {
   wants_box(x); //~ ERROR str storage differs: expected @ but found &
   wants_uniq(x); //~ ERROR str storage differs: expected ~ but found &
   wants_slice(x);
}

fn main() {
}
