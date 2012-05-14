enum maybe<T> { just(T), nothing }

impl methods<T:copy> for maybe<T> {
    fn [](idx: uint) -> T {
        alt self {
          just(t) { t }
          nothing { fail; }
        }
    }
}