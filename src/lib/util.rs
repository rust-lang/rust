type option[T] = tag(none(), some(T));
type box[T] = tup(@T);
type boxo[T] = option[box[T]];
type boxm[T] = tup(mutable @T);
type boxmo[T] = option[boxm[T]];

type map[T, U] = fn(&T) -> U;

fn option_map[T, U](map[T, U] f, &option[T] opt) -> option[U] {
  alt (opt) {
    case (some[T](x)) {
      ret some[U](f[T, U](x));
    }
    case (none[T]()) {
      ret none[U]();
    }
  }
}

fn unbox[T](&box[T] b) -> T {
  ret b._0;
}


fn unboxm[T](&boxm[T] b) -> T {
  ret b._0;
}

fn unboxo[T](boxo[T] b) -> option[T] {
  // Pending issue #90, no need to alias the function item in order to pass
  // it as an arg.
  let map[box[T], T] f = unbox[T];
  be option_map[box[T], T](f, b);
}

fn unboxmo[T](boxmo[T] b) -> option[T] {
  // Issue #90, as above
  let map[boxm[T], T] f = unboxm[T];
  be option_map[boxm[T], T](f, b);
}

fn id[T](T x) -> T {
  ret x;
}

type rational = rec(int num, int den);
