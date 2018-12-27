// compile-flags: -Z parse-only

trait MyTrait<T>: Iterator { //~ ERROR missing `fn`, `type`, or `const`
    Item = T;
}
