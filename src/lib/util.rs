type option[T] = tag(none(), some(T));

type operator[T, U] = fn(&T) -> U;

fn option_map[T, U](&operator[T, U] f, &option[T] opt) -> option[U] {
  alt (opt) {
    case (some[T](x)) {
      ret some[U](f[T, U](x));
    }
    case (none[T]()) {
      ret none[U]();
    }
  }
}

fn id[T](T x) -> T {
  ret x;
}

type rational = rec(int num, int den);
