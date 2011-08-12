

fn mk() -> int { ret 1; }

fn chk(a: &int) { log a; assert (a == 1); }

fn apply<T>(produce: fn() -> T , consume: fn(&T) ) { consume(produce()); }

fn main() {
    let produce: fn() -> int  = mk;
    let consume: fn(&int)  = chk;
    apply[int](produce, consume);
}
