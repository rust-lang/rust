

fn mk() -> int { ret 1; }

fn chk(&int a) { log a; assert (a == 1); }

fn apply[T](fn() -> T  produce, fn(&T)  consume) { consume(produce()); }

fn main() {
    let fn() -> int  produce = mk;
    let fn(&int)  consume = chk;
    apply[int](produce, consume);
}