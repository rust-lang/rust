fn wants_box(x: str/@) { }
fn wants_uniq(x: str/~) { }
fn wants_three(x: str/3) { }

fn has_box(x: str/@) {
   wants_box(x);
   wants_uniq(x); //~ ERROR str storage differs: expected ~ but found @
   wants_three(x); //~ ERROR str storage differs: expected 3 but found @
}

fn has_uniq(x: str/~) {
   wants_box(x); //~ ERROR str storage differs: expected @ but found ~
   wants_uniq(x);
   wants_three(x); //~ ERROR str storage differs: expected 3 but found ~
}

fn has_three(x: str/3) {
   wants_box(x); //~ ERROR str storage differs: expected @ but found 3
   wants_uniq(x); //~ ERROR str storage differs: expected ~ but found 3
   wants_three(x);
}

fn has_four(x: str/4) {
   wants_box(x); //~ ERROR str storage differs: expected @ but found 4
   wants_uniq(x); //~ ERROR str storage differs: expected ~ but found 4
   wants_three(x); //~ ERROR str storage differs: expected 3 but found 4
}

fn main() {
}
