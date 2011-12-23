


// -*- rust -*-
tag clam<T> { a(T, int); b; }

fn uhoh<T>(v: [clam<T>]) {
    alt v[1] {
      a::<T>(t, u) { #debug("incorrect"); log(debug, u); fail; }
      b::<T>. { #debug("correct"); }
    }
}

fn main() {
    let v: [clam<int>] = [b::<int>, b::<int>, a::<int>(42, 17)];
    uhoh::<int>(v);
}
