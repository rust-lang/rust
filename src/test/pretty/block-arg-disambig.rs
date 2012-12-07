fn blk1(b: fn()) -> fn@() { return fn@() { }; }
fn test1() { (do blk1 { debug!("hi"); })(); }
