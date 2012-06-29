use issue2378a;

import issue2378a::maybe;
import issue2378a::methods;

type two_maybes<T> = {a: maybe<T>, b: maybe<T>};

impl methods<T:copy> for two_maybes<T> {
    fn ~[](idx: uint) -> (T, T) {
        (self.a[idx], self.b[idx])
    }
}